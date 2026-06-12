"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model Qwen/Qwen3.5-35B-A3B \
	--mem-fraction-static 0.7 \
	--tp 1 \
	--trust-remote-code \
    --cuda-graph-max-bs 128 \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16 \
    --reasoning-parser qwen3


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 localhost:30040 localhost:30050 localhost:30060 localhost:30070 \
    --temperature 0.8 \
    --input-file-path /data/jiapingW/pr/SpecForge/cache/dataset/opc_train_first_turn.jsonl \
    --output-file-path ./cache/dataset/opc_train_regen_first_turn.jsonl \
    --resume \
    --reasoning save

For VLM JSONL rows with top-level image fields, add --is-vlm. The script scans
image, images, image_path, image_file, and image_url by default, then injects
the image(s) into the first user message as OpenAI-compatible image_url content.
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


IMAGE_FIELD_NAMES = ("image", "images", "image_path", "image_file", "image_url")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    # model related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--reasoning",
        choices=["none", "save", "disable"],
        default="none",
        help=(
            "Reasoning mode: 'none' for standard models, 'save' to store "
            "reasoning_content, or 'disable' to disable thinking via extra_body"
        ),
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    model_group.add_argument(
        "--is-vlm",
        action="store_true",
        help=(
            "Whether to inject top-level image fields into user messages as "
            "OpenAI-compatible image_url content."
        ),
    )
    model_group.add_argument(
        "--image-field-names",
        type=str,
        nargs="+",
        default=list(IMAGE_FIELD_NAMES),
        help="Top-level JSONL fields to scan for VLM image paths or URLs.",
    )

    # sampling params
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of output tokens (default: 4096)",
    )

    # optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )

    # data related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    data_group.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skip already processed samples",
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """
    This is a rough estimate of the context length measured in untokenized
    tokens.
    """
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            # {"role": "assistant", "content": "Hi, how can I help?"}
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def _extract_image_url_from_dict(value: Dict[str, Any]) -> Optional[str]:
    if "url" in value and value["url"]:
        return str(value["url"])
    image_url = value.get("image_url")
    if isinstance(image_url, dict) and image_url.get("url"):
        return str(image_url["url"])
    if isinstance(image_url, str) and image_url:
        return image_url
    for key in ("image", "image_path", "image_file", "path"):
        if value.get(key):
            return str(value[key])
    return None


def _normalize_image_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, dict):
        image_url = _extract_image_url_from_dict(value)
        return [image_url] if image_url else []
    if isinstance(value, (list, tuple)):
        image_urls = []
        for item in value:
            image_urls.extend(_normalize_image_values(item))
        return image_urls
    return [str(value)]


def get_image_urls(data: Dict[str, Any], image_field_names: List[str]) -> List[str]:
    image_urls = []
    for field_name in image_field_names:
        if field_name not in data:
            continue
        image_urls.extend(_normalize_image_values(data[field_name]))
        if image_urls:
            break
    return image_urls


def _is_image_content_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in {"image", "image_url"}


def conversations_have_images(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            _is_image_content_part(part) for part in content
        ):
            return True
    return False


def _normalize_content_part(part: Any) -> Optional[Dict[str, Any]]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    part_type = part.get("type")
    if part_type == "text":
        return {"type": "text", "text": str(part.get("text", ""))}
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            detail = image_url.get("detail")
        else:
            url = image_url
            detail = None
        if not url:
            return None
        normalized = {"type": "image_url", "image_url": {"url": str(url)}}
        if detail:
            normalized["image_url"]["detail"] = detail
        return normalized
    if part_type == "image":
        url = part.get("image") or part.get("url")
        if not url:
            return None
        return {"type": "image_url", "image_url": {"url": str(url)}}

    return {"type": "text", "text": str(part.get("text", part))}


def normalize_openai_message(message: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {k: v for k, v in message.items() if k != "content"}
    content = message.get("content", "")
    if isinstance(content, list):
        normalized_parts = [
            normalized_part
            for part in content
            if (normalized_part := _normalize_content_part(part)) is not None
        ]
        normalized["content"] = normalized_parts
    else:
        normalized["content"] = content
    return normalized


def inject_images_into_user_message(
    message: Dict[str, Any], image_urls: List[str]
) -> Dict[str, Any]:
    content = message.get("content", "")
    image_parts = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    if isinstance(content, list):
        content_parts = image_parts + content
    elif content:
        content_parts = image_parts + [{"type": "text", "text": str(content)}]
    else:
        content_parts = image_parts

    injected = dict(message)
    injected["content"] = content_parts
    return injected


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.reasoning == "disable":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def build_error_record(
    data: Dict[str, Any],
    error: str,
    failed_turn_index: Optional[int] = None,
    failed_message_index: Optional[int] = None,
    regenerated_turns: int = 0,
    partial_success_written: bool = False,
) -> Dict[str, Any]:
    error_record = dict(data)
    error_record["status"] = "error"
    error_record["error"] = error
    error_record["regenerated_turns"] = regenerated_turns
    error_record["partial_success_written"] = partial_success_written
    if failed_turn_index is not None:
        error_record["failed_turn_index"] = failed_turn_index
    if failed_message_index is not None:
        error_record["failed_message_index"] = failed_message_index
    return error_record


def build_success_record(
    data: Dict[str, Any],
    conversations: List[Dict[str, Any]],
    status: str,
    regenerated_turns: int,
    failed_turn_index: Optional[int] = None,
    failed_message_index: Optional[int] = None,
) -> Dict[str, Any]:
    success_record = dict(data)
    success_record["conversations"] = conversations
    success_record["status"] = status
    success_record["regeneration_status"] = status
    success_record["regenerated_turns"] = regenerated_turns
    if failed_turn_index is not None:
        success_record["failed_turn_index"] = failed_turn_index
    if failed_message_index is not None:
        success_record["failed_message_index"] = failed_message_index
    return success_record


def build_regeneration_result(
    status: str,
    data: Optional[Dict[str, Any]] = None,
    error_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {"status": status}
    if data is not None:
        result["data"] = data
    if error_data is not None:
        result["error_data"] = error_data
    return result


def count_resume_error_samples(error_file_path: str) -> int:
    """
    Count error-only input rows for resume.

    Partial successes write both one output row and one error record. The output
    row already accounts for that input row, so those error records must not
    increase the resume skip count.
    """
    if not os.path.exists(error_file_path):
        return 0

    count = 0
    with open(error_file_path, "r") as error_file:
        for line in error_file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                count += 1
                continue
            if record.get("partial_success_written"):
                continue
            count += 1
    return count


def call_sglang(
    args,
    server_address: str,
    data: Dict[str, Any],
    max_tokens=None,
) -> Dict[str, Any]:
    """Send a batch of prompts to sglang /v1/completions."""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []
    image_urls = get_image_urls(data, args.image_field_names) if args.is_vlm else []
    should_inject_images = (
        args.is_vlm and image_urls and not conversations_have_images(messages)
    )
    injected_images = False

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        error_record = build_error_record(
            data,
            "Data starts with an assistant message",
            failed_turn_index=0,
            failed_message_index=0,
        )
        return build_regeneration_result("error", error_data=error_record)

    regenerated_turns = 0
    user_turn_index = 0
    for message_index, message in enumerate(messages):
        message = normalize_openai_message(message)
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            user_turn_index += 1
            prefix_len_before_user = len(regenerated_messages)
            if should_inject_images and not injected_images:
                message = inject_images_into_user_message(message, image_urls)
                injected_images = True
            regenerated_messages.append(message)

            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                error_record = build_error_record(
                    data,
                    str(e),
                    failed_turn_index=user_turn_index,
                    failed_message_index=message_index,
                    regenerated_turns=regenerated_turns,
                    partial_success_written=regenerated_turns > 0,
                )
                if regenerated_turns > 0:
                    success_record = build_success_record(
                        data,
                        regenerated_messages[:prefix_len_before_user],
                        "partial_success",
                        regenerated_turns,
                        failed_turn_index=user_turn_index,
                        failed_message_index=message_index,
                    )
                    return build_regeneration_result(
                        "partial_success",
                        data=success_record,
                        error_data=error_record,
                    )
                return build_regeneration_result("error", error_data=error_record)
            response_text = resp.choices[0].message.content
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.reasoning == "save":
                resp_msg["reasoning_content"] = resp.choices[
                    0
                ].message.reasoning_content
            regenerated_messages.append(resp_msg)
            regenerated_turns += 1
        else:
            error_record = build_error_record(
                data,
                f"Invalid message role: {message['role']}",
                failed_turn_index=user_turn_index + 1,
                failed_message_index=message_index,
                regenerated_turns=regenerated_turns,
                partial_success_written=regenerated_turns > 0,
            )
            if regenerated_turns > 0:
                success_record = build_success_record(
                    data,
                    regenerated_messages,
                    "partial_success",
                    regenerated_turns,
                    failed_turn_index=user_turn_index + 1,
                    failed_message_index=message_index,
                )
                return build_regeneration_result(
                    "partial_success", data=success_record, error_data=error_record
                )
            return build_regeneration_result("error", error_data=error_record)
    success_record = build_success_record(
        data,
        regenerated_messages,
        "success",
        regenerated_turns,
    )
    return build_regeneration_result("success", data=success_record)


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print("-" * 50)
    total_lines = sum(1 for _ in open(args.input_file_path))

    skip_lines = 0
    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")

    if args.resume and os.path.exists(args.output_file_path):
        existing_success = sum(1 for _ in open(args.output_file_path))
        existing_error = count_resume_error_samples(error_file_path)
        skip_lines = existing_success + existing_error
        print(f"Resume mode enabled:")
        print(f"  Found {existing_success} successful samples in output file")
        print(f"  Found {existing_error} error-only samples in error file")
        print(f"  Skipping first {skip_lines} input samples")
        print("-" * 50)

        if skip_lines >= total_lines:
            print(f"All {total_lines} samples already processed. Nothing to do.")
            return

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(
            args,
            server_address,
            dummy_data,
            max_tokens=1,
        )
        if result is not None:
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    # Determine file open mode based on resume flag
    file_mode = "a" if (args.resume and skip_lines > 0) else "w"
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print(
        f"File open mode: {file_mode} ({'append' if file_mode == 'a' else 'overwrite'})"
    )
    print("-" * 50)
    context_token_sum = 0
    context_token_min = None
    context_token_max = 0
    success_samples = 0
    error_samples = 0
    partial_samples = 0
    submitted_samples = 0

    # Create progress bar
    with (
        open(args.input_file_path, "r") as input_file,
        open(args.output_file_path, file_mode) as output_file_handle,
        open(error_file_path, file_mode) as error_file_handle,
    ):
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        pbar = tqdm(total=total_lines, desc="Processing", initial=skip_lines)
        start_server_index = 0

        def write_regeneration_result(regen_result: Dict[str, Any]) -> None:
            nonlocal context_token_sum
            nonlocal context_token_min
            nonlocal context_token_max
            nonlocal success_samples
            nonlocal error_samples
            nonlocal partial_samples

            status = regen_result["status"]
            success_data = regen_result.get("data")
            error_data = regen_result.get("error_data")

            if error_data is not None:
                error_file_handle.write(
                    json.dumps(error_data, ensure_ascii=False) + "\n"
                )
                error_samples += 1

            if success_data is None:
                return

            ctx_len = compute_context_length(success_data.get("conversations", []))
            context_token_sum += ctx_len
            if context_token_min is None:
                context_token_min = ctx_len
            else:
                context_token_min = min(context_token_min, ctx_len)
            context_token_max = max(context_token_max, ctx_len)

            output_file_handle.write(
                json.dumps(success_data, ensure_ascii=False) + "\n"
            )
            success_samples += 1
            if status == "partial_success":
                partial_samples += 1

        if skip_lines > 0:
            print(f"Skipping {skip_lines} already processed samples...")
            for _ in range(skip_lines):
                next(input_file, None)
            print(f"Resuming from sample {skip_lines + 1}")

        for line in input_file:
            if args.num_samples is not None and submitted_samples >= args.num_samples:
                break

            data = json.loads(line.strip())

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        write_regeneration_result(req_future.result())
                        waiting_queue[server_address].remove(req_future)
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args,
                server_address,
                data,
            )
            waiting_queue[server_address].append(req_future)
            submitted_samples += 1
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_future in waiting_queue_items:
                write_regeneration_result(req_future.result())

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = context_token_sum / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {context_token_min}")
        print(f"Longest context length: {context_token_max}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    if skip_lines > 0:
        print(f"\nResume processing completed!")
        print(f"  Previously processed: {skip_lines}")
        print(
            f"  Newly processed input rows: {submitted_samples}"
            f" ({success_samples} output rows, {partial_samples} partial, {error_samples} error records)"
        )
        print(f"  Total input rows accounted: {skip_lines + submitted_samples}")
    else:
        print(
            f"\nProcessing completed! {submitted_samples} input rows processed, "
            f"{success_samples} output rows, {partial_samples} partial successes, "
            f"{error_samples} error records."
        )


if __name__ == "__main__":
    main()
