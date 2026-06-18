import os
import time
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from torch import nn
from transformers import PretrainedConfig
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.eagle3 import OnlineEagle3Model
from specforge.core.eagle3_adapters import SdpaLikeAdapter, UspAdapter
from specforge.data.preprocessing import (
    OfflineEagle3Dataset,
    build_offline_eagle3_dataset,
)
from specforge.data.utils import DataCollatorWithPadding

# Project-specific imports
from specforge.distributed import (
    destroy_distributed,
    get_draft_sp_group,
    get_sp_ring_group,
    get_sp_ulysses_group,
    init_distributed,
)
from specforge.modeling.draft.llama3_eagle import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
)
from specforge.utils import padding
from tests.utils import get_available_port


def get_model_config():
    """Create and return the model configuration."""
    config_dict = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 29, 57],
            "use_aux_hidden_state": True,
        },
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 29568,
        "max_position_embeddings": 32768,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.28.1",
        "use_cache": True,
        "rope_scaling": None,
        "vocab_size": 129280,
        "draft_vocab_size": 32000,
        "pretraining_tp": 1,
    }
    return PretrainedConfig.from_dict(config_dict)


def get_mrope_model_config():
    """Create a compact Qwen-VL-style MRoPE decoder config for parity tests."""
    config_dict = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": 1024,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "type": "mrope",
            "mrope_section": [2, 3, 3],
        },
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 1024,
        "draft_vocab_size": 1024,
    }
    return PretrainedConfig.from_dict(config_dict)


def make_mrope_position_ids(seq_len, device):
    base = torch.arange(seq_len, device=device)
    temporal = base.clone()
    height = base.clone()
    width = base.clone()

    vision_len = seq_len // 4
    vision_positions = torch.arange(vision_len, device=device)
    temporal[:vision_len] = 0
    height[:vision_len] = vision_positions // 8
    width[:vision_len] = vision_positions % 8

    return torch.stack([temporal, height, width], dim=0).unsqueeze(1).long()


def reduced_named_grads(module):
    grads = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}"
        grad = parameter.grad.detach().float().clone()
        torch.distributed.all_reduce(grad)
        grads[name] = grad
    return grads


def assert_reduced_grads_close(fa_decoder, usp_decoder, rank, label):
    fa_grads = reduced_named_grads(fa_decoder)
    usp_grads = reduced_named_grads(usp_decoder)
    assert fa_grads.keys() == usp_grads.keys()
    for name, fa_grad in fa_grads.items():
        usp_grad = usp_grads[name]
        max_diff = (usp_grad - fa_grad).abs().max().item()
        assert torch.allclose(usp_grad, fa_grad, rtol=5e-2, atol=5e-2), (
            f"[Rank {rank}] MRoPE USP backward {label} grad mismatch for {name}; "
            f"max diff={max_diff}"
        )


def make_offline_mrope_sample(config, seq_len, device):
    return {
        "input_ids": torch.randint(
            0, config.vocab_size, (seq_len,), device=device, dtype=torch.long
        ),
        "loss_mask": torch.ones(seq_len, device=device, dtype=torch.long),
        "hidden_state": torch.randn(
            1, seq_len, config.draft_vocab_size, device=device, dtype=torch.bfloat16
        ),
        "aux_hidden_state": torch.randn(
            1, seq_len, config.hidden_size * 3, device=device, dtype=torch.bfloat16
        ),
        "position_ids": make_mrope_position_ids(seq_len, device),
    }


def move_training_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def assert_model_backward_ran(draft_model):
    grad = draft_model.midlayer.self_attn.q_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0


def setup_env(rank, world_size, port):
    """Set up distributed environment variables."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)


def dbg(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def wait_for_file(path, timeout_s=60, poll_s=0.1):
    start = time.time()
    while time.time() - start < timeout_s:
        if os.path.exists(path):
            return True
        time.sleep(poll_s)
    return False


def run_iterative_pass(
    decoder_layer,
    embed_tokens,
    input_ids,
    hidden_states,
    attention_mask,
    position_ids,
    ttt_length,
):
    """
    Core loop: execute the forward pass `ttt_length` times.
    Used for both Golden (SDPA) and Distributed (USP) runs to ensure logic consistency.
    """
    # Clone to avoid side effects on original tensors
    curr_input_ids = input_ids.clone()
    curr_hidden_states = hidden_states.clone()

    # Init cache
    cache_hidden = [[], []]
    past_key_values = None
    final_output = None

    for idx in range(ttt_length):
        is_last = idx == ttt_length - 1

        # 1. Embed inputs
        inputs_embeds = embed_tokens(curr_input_ids).to(curr_hidden_states.dtype)

        # 2. Forward pass
        output_hidden_states = decoder_layer(
            input_emb=inputs_embeds,
            hidden_states=curr_hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

        # Update states for next iteration
        curr_hidden_states = output_hidden_states
        final_output = output_hidden_states

        # 3. Simulate TTT padding/shift
        if not is_last:
            curr_input_ids = padding(curr_input_ids, left=False)

    return final_output


def run_test_case(rank, world_size, port):
    """Worker function executed in each process."""
    setup_env(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    set_seed(42)
    dbg(rank, "env setup complete")

    # --- Data & Config Preparation ---
    config = get_model_config()
    seq_len = 1560
    batch_size = 1
    ttt_length = 3

    # Generate dummy data on GPU
    data_input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    data_hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Shared embedding layer
    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id
    ).to(device)

    # --- Phase 1: Golden Run (FA) ---
    # Init dist briefly for internal checks, even if running single-device logic
    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    dbg(rank, "init_distributed (FA) done")

    sdpa_decoder = (
        LlamaDecoderLayer(config, attention_backend="fa").to(device).to(torch.bfloat16)
    )
    dbg(rank, "FA decoder created")
    # Adapter smoke test for FA/SDPA-style path
    dummy_model = type("Dummy", (), {})()
    sdpa_adapter = SdpaLikeAdapter(dummy_model)
    sdpa_target_p = torch.zeros((1, seq_len, 8), device=device, dtype=torch.float32)
    sdpa_position_mask = torch.ones((1, seq_len, 1), device=device, dtype=torch.float32)
    sdpa_state = sdpa_adapter.step_view(
        idx=0,
        ttt_length=ttt_length,
        global_input_ids=data_input_ids,
        attention_mask=attention_mask,
        loss_mask=torch.ones((1, seq_len, 1), device=device, dtype=torch.float32),
        position_ids=position_ids,
        hidden_states=data_hidden_states,
        target_p_padded=sdpa_target_p,
        position_mask=sdpa_position_mask,
        seq_length=seq_len,
    )
    assert sdpa_state.input_ids.shape[1] == seq_len
    assert sdpa_state.hidden_states.shape[1] == seq_len

    with torch.no_grad():
        sdpa_output = run_iterative_pass(
            decoder_layer=sdpa_decoder,
            embed_tokens=embed_tokens,
            input_ids=data_input_ids,
            hidden_states=data_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ttt_length=ttt_length,
        )
    dbg(rank, "FA forward done")

    # Save weights for alignment and cleanup SDPA model
    state_dict = sdpa_decoder.state_dict()
    del sdpa_decoder
    destroy_distributed()
    dbg(rank, "destroy_distributed (FA) done")

    # --- Phase 2: Distributed Run (USP) ---
    def subtest_usp(sp_ulysses_degree, sp_ring_degree):
        """Run USP with specific topology and compare against Golden."""
        try:
            init_distributed(
                tp_size=1,
                sp_ulysses_size=sp_ulysses_degree,
                sp_ring_size=sp_ring_degree,
            )
            dbg(
                rank,
                f"init_distributed (USP U{sp_ulysses_degree} R{sp_ring_degree}) done",
            )
            # Dataset + adapter smoke test (USP path)
            tmp_dir = "./tmp/usp_dataset_shared"
            try:
                if rank == 0:
                    os.makedirs(tmp_dir, exist_ok=True)
                    sample_mrope_position_ids = torch.stack(
                        [
                            torch.arange(seq_len),
                            torch.arange(seq_len) + 10_000,
                            torch.arange(seq_len) + 20_000,
                        ],
                        dim=0,
                    )
                    sample = {
                        "input_ids": data_input_ids[0].cpu(),
                        "loss_mask": torch.ones_like(data_input_ids[0].cpu()),
                        "hidden_state": data_hidden_states[0].cpu().unsqueeze(0),
                        "aux_hidden_state": data_hidden_states[0].cpu().unsqueeze(0),
                        "position_ids": sample_mrope_position_ids,
                    }
                    torch.save(sample, os.path.join(tmp_dir, "data_0.ckpt"))
                    dbg(rank, "wrote sample ckpt")
                    ready_flag = os.path.join(tmp_dir, "ready.flag")
                    with open(ready_flag, "w", encoding="utf-8") as f:
                        f.write("ready\n")
                if rank != 0:
                    ready_flag = os.path.join(tmp_dir, "ready.flag")
                    assert wait_for_file(
                        ready_flag, timeout_s=60
                    ), "timeout waiting for ready flag"
                dbg(rank, "dataset sync done")
                assert os.path.exists(
                    os.path.join(tmp_dir, "data_0.ckpt")
                ), f"Expected sample not found at {tmp_dir}"
                dbg(rank, "sample exists")

                ds = build_offline_eagle3_dataset(
                    tmp_dir,
                    max_len=seq_len,
                    ttt_length=ttt_length,
                    use_usp_preprocess=True,
                )
                dbg(rank, "dataset built")
                item = ds[0]
                dbg(rank, "dataset item loaded")
                assert "position_ids" in item

                dummy_model = type("Dummy", (), {})()
                adapter = UspAdapter(dummy_model)
                local_seq_len = item["input_ids"].shape[1]
                target_p_padded = torch.zeros(
                    (1, local_seq_len, 8), device=device, dtype=torch.float32
                )
                position_mask = torch.ones(
                    (1, local_seq_len, 1), device=device, dtype=torch.float32
                )
                state = adapter.step_view(
                    idx=0,
                    ttt_length=ttt_length,
                    global_input_ids=item["input_ids"].to(device),
                    attention_mask=item["attention_mask"].to(device),
                    loss_mask=item["loss_mask"].to(device).unsqueeze(-1),
                    position_ids=item["position_ids"].to(device),
                    hidden_states=item["hidden_state"].to(device),
                    target_p_padded=target_p_padded,
                    position_mask=position_mask,
                    seq_length=local_seq_len,
                )
                assert state.input_ids.shape[1] == local_seq_len - ttt_length
                assert state.hidden_states.shape[1] == local_seq_len - ttt_length
                assert state.position_ids is not None
                expected_position_len = (
                    local_seq_len - ttt_length
                ) * sp_ulysses_degree
                assert state.position_ids.shape == (3, 1, expected_position_len)
                assert torch.equal(
                    torch.diff(state.position_ids[0, 0].cpu()),
                    torch.ones(expected_position_len - 1, dtype=torch.long),
                )
                dbg(rank, "adapter step_view ok")
            finally:
                if rank == 0:
                    done_flag = os.path.join(tmp_dir, "done.flag")
                    assert wait_for_file(
                        done_flag, timeout_s=60
                    ), "timeout waiting for done flag"
                    try:
                        for root, _, files in os.walk(tmp_dir):
                            for name in files:
                                os.remove(os.path.join(root, name))
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass
                else:
                    done_flag = os.path.join(tmp_dir, "done.flag")
                    with open(done_flag, "w", encoding="utf-8") as f:
                        f.write("done\n")

            # Init USP model and load golden weights
            usp_decoder = (
                LlamaDecoderLayer(config, attention_backend="usp")
                .to(device)
                .to(torch.bfloat16)
            )
            usp_decoder.load_state_dict(state_dict)
            dbg(rank, "USP decoder loaded")

            # Shard data (Split Input)
            extract_func = EXTRACT_FUNC_DICT["basic"]

            local_input_ids = (
                extract_func(
                    data_input_ids,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )

            local_hidden_states = (
                extract_func(
                    data_hidden_states,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )
            dbg(rank, "USP local inputs prepared")
            total_degree = sp_ring_degree * sp_ulysses_degree
            chunk_size = sdpa_output.shape[1] // total_degree
            start_idx = (rank % total_degree) * chunk_size
            local_len = local_input_ids.shape[1]
            local_position_ids = (
                torch.arange(start_idx, start_idx + local_len, device=device)
                .unsqueeze(0)
                .long()
            )
            local_attention_mask = torch.tril(
                torch.ones(local_len, local_len, device=device)
            ).view(1, 1, local_len, local_len)

            # Run USP forward
            if sp_ring_degree > 1:
                usp_attention_mask = local_attention_mask
                usp_position_ids = local_position_ids
            else:
                usp_attention_mask = attention_mask
                usp_position_ids = position_ids
            with torch.no_grad():
                usp_output = run_iterative_pass(
                    decoder_layer=usp_decoder,
                    embed_tokens=embed_tokens,
                    input_ids=local_input_ids,
                    hidden_states=local_hidden_states,
                    attention_mask=usp_attention_mask,
                    position_ids=usp_position_ids,
                    ttt_length=ttt_length,
                )
            dbg(rank, "USP forward done")

            # Verify results
            # Slice the golden output to match the current rank's chunk
            end_idx = start_idx + chunk_size

            golden_chunk = sdpa_output[:, start_idx:end_idx, :]

            assert torch.allclose(usp_output, golden_chunk, rtol=2e-2, atol=2e-2), (
                f"[Rank {rank}] USP (U{sp_ulysses_degree}R{sp_ring_degree}) mismatch!\n"
                f"Max Diff: {(usp_output - golden_chunk).abs().max().item()}"
            )
            dbg(rank, "USP output verified")

        finally:
            destroy_distributed()
            dbg(rank, "destroy_distributed (USP) done")

    # Case 1: Hybrid (Ulysses=2, Ring=1)
    subtest_usp(sp_ulysses_degree=2, sp_ring_degree=1)

    # Case 2: Hybrid (Ulysses=1, Ring=2)
    subtest_usp(sp_ulysses_degree=1, sp_ring_degree=2)


def run_mrope_parity_case(rank, world_size, port):
    """Compare MRoPE USP decoder output against a full-sequence FA baseline."""
    setup_env(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    set_seed(1234)
    dbg(rank, "MRoPE parity env setup complete")

    config = get_mrope_model_config()
    seq_len = 256
    batch_size = 1
    ttt_length = 1

    data_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    data_hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )
    position_ids = make_mrope_position_ids(seq_len, device)
    assert not torch.equal(position_ids[0], position_ids[1])
    assert not torch.equal(position_ids[1], position_ids[2])

    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, config.pad_token_id
    ).to(device)

    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    fa_decoder = (
        LlamaDecoderLayer(config, attention_backend="fa").to(device).to(torch.bfloat16)
    )
    with torch.no_grad():
        fa_output = run_iterative_pass(
            decoder_layer=fa_decoder,
            embed_tokens=embed_tokens,
            input_ids=data_input_ids,
            hidden_states=data_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ttt_length=ttt_length,
        )
    state_dict = fa_decoder.state_dict()
    del fa_decoder
    destroy_distributed()
    dbg(rank, "MRoPE FA baseline done")

    def subtest_mrope_usp(sp_ulysses_degree, sp_ring_degree):
        try:
            init_distributed(
                tp_size=1,
                sp_ulysses_size=sp_ulysses_degree,
                sp_ring_size=sp_ring_degree,
            )
            usp_decoder = (
                LlamaDecoderLayer(config, attention_backend="usp")
                .to(device)
                .to(torch.bfloat16)
            )
            usp_decoder.load_state_dict(state_dict)

            extract_func = EXTRACT_FUNC_DICT["basic"]
            local_input_ids = (
                extract_func(
                    data_input_ids,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )
            local_hidden_states = (
                extract_func(
                    data_hidden_states,
                    rank,
                    world_size=world_size,
                    rd=sp_ring_degree,
                    ud=sp_ulysses_degree,
                )
                .detach()
                .clone()
            )

            total_degree = sp_ring_degree * sp_ulysses_degree
            chunk_size = fa_output.shape[1] // total_degree
            start_idx = (rank % total_degree) * chunk_size
            local_len = local_input_ids.shape[1]
            local_position_ids = position_ids[..., start_idx : start_idx + local_len]
            if sp_ulysses_degree > 1:
                usp_position_ids = UspAdapter._all_gather_seq_last(
                    local_position_ids, get_sp_ulysses_group()
                )
            else:
                usp_position_ids = local_position_ids

            expected_position_len = local_len * sp_ulysses_degree
            assert usp_position_ids.shape == (3, batch_size, expected_position_len)

            local_attention_mask = torch.tril(
                torch.ones(local_len, local_len, device=device)
            ).view(1, 1, local_len, local_len)
            usp_attention_mask = (
                local_attention_mask if sp_ring_degree > 1 else attention_mask
            )

            with torch.no_grad():
                usp_output = run_iterative_pass(
                    decoder_layer=usp_decoder,
                    embed_tokens=embed_tokens,
                    input_ids=local_input_ids,
                    hidden_states=local_hidden_states,
                    attention_mask=usp_attention_mask,
                    position_ids=usp_position_ids,
                    ttt_length=ttt_length,
                )

            golden_chunk = fa_output[:, start_idx : start_idx + chunk_size, :]
            max_diff = (usp_output - golden_chunk).abs().max().item()
            assert torch.allclose(
                usp_output, golden_chunk, rtol=3e-2, atol=3e-2
            ), (
                f"[Rank {rank}] MRoPE USP "
                f"(U{sp_ulysses_degree}R{sp_ring_degree}) mismatch; "
                f"max diff={max_diff}"
            )
            dbg(
                rank,
                f"MRoPE USP U{sp_ulysses_degree}R{sp_ring_degree} verified "
                f"with max diff {max_diff:.6f}",
            )

            fa_grad_decoder = (
                LlamaDecoderLayer(config, attention_backend="fa")
                .to(device)
                .to(torch.bfloat16)
            )
            fa_grad_decoder.load_state_dict(state_dict)
            usp_grad_decoder = (
                LlamaDecoderLayer(config, attention_backend="usp")
                .to(device)
                .to(torch.bfloat16)
            )
            usp_grad_decoder.load_state_dict(state_dict)

            fa_grad_output = run_iterative_pass(
                decoder_layer=fa_grad_decoder,
                embed_tokens=embed_tokens,
                input_ids=data_input_ids,
                hidden_states=data_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                ttt_length=ttt_length,
            )
            fa_grad_chunk = fa_grad_output[:, start_idx : start_idx + chunk_size, :]
            fa_loss = fa_grad_chunk.float().pow(2).mean()
            fa_loss.backward()

            usp_grad_output = run_iterative_pass(
                decoder_layer=usp_grad_decoder,
                embed_tokens=embed_tokens,
                input_ids=local_input_ids,
                hidden_states=local_hidden_states,
                attention_mask=usp_attention_mask,
                position_ids=usp_position_ids,
                ttt_length=ttt_length,
            )
            usp_loss = usp_grad_output.float().pow(2).mean()
            usp_loss.backward()

            label = f"U{sp_ulysses_degree}R{sp_ring_degree}"
            assert_reduced_grads_close(
                fa_grad_decoder, usp_grad_decoder, rank, label
            )
            dbg(rank, f"MRoPE USP {label} backward gradients verified")
        finally:
            destroy_distributed()

    subtest_mrope_usp(sp_ulysses_degree=2, sp_ring_degree=1)
    subtest_mrope_usp(sp_ulysses_degree=1, sp_ring_degree=2)


def run_mrope_offline_full_flow_fa_case(rank, world_size, port):
    setup_env(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    set_seed(2026)
    init_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
    try:
        config = get_mrope_model_config()
        seq_len = 64
        item = OfflineEagle3Dataset.process_data(
            make_offline_mrope_sample(config, seq_len, torch.device("cpu")),
            max_len=seq_len,
        )
        batch = move_training_batch_to_device(DataCollatorWithPadding()([item]), device)

        draft_model = (
            LlamaForCausalLMEagle3(config, attention_backend="fa")
            .to(device)
            .to(torch.bfloat16)
        )
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model,
            length=1,
            attention_backend="fa",
        )

        plosses, *_ = eagle3_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"].unsqueeze(-1),
            target=batch["target"],
            hidden_states=batch["hidden_state"],
            position_ids=batch["position_ids"],
            is_vlm=True,
        )
        assert len(plosses) == 1
        assert torch.isfinite(plosses[0])
        plosses[0].backward()
        assert_model_backward_ran(draft_model)
        dbg(rank, "MRoPE offline non-SP full flow backward verified")
    finally:
        destroy_distributed()


def run_mrope_offline_full_flow_usp_case(rank, world_size, port):
    setup_env(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    set_seed(2027)
    init_distributed(tp_size=1, sp_ulysses_size=2, sp_ring_size=1)
    try:
        config = get_mrope_model_config()
        seq_len = 128
        sp_group = get_draft_sp_group()
        ring_group = get_sp_ring_group()
        item = OfflineEagle3Dataset.process_data_usp(
            make_offline_mrope_sample(config, seq_len, torch.device("cpu")),
            max_len=seq_len,
            ttt_length=1,
            sp_rank=torch.distributed.get_rank(sp_group),
            sp_size=torch.distributed.get_world_size(sp_group),
            ring_rank=torch.distributed.get_rank(ring_group),
            sp_ring_size=torch.distributed.get_world_size(ring_group),
        )
        batch = move_training_batch_to_device(DataCollatorWithPadding()([item]), device)

        draft_model = (
            LlamaForCausalLMEagle3(config, attention_backend="usp")
            .to(device)
            .to(torch.bfloat16)
        )
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model,
            length=1,
            attention_backend="usp",
        )

        plosses, *_ = eagle3_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"].unsqueeze(-1),
            target=batch["target"],
            hidden_states=batch["hidden_state"],
            position_ids=batch["position_ids"],
            is_vlm=True,
        )
        assert len(plosses) == 1
        assert torch.isfinite(plosses[0])
        plosses[0].backward()
        assert_model_backward_ran(draft_model)
        dbg(rank, "MRoPE offline USP full flow backward verified")
    finally:
        destroy_distributed()


class TestTTTDistributed(unittest.TestCase):
    def test_llama_usp_decoder(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(run_test_case, nprocs=world_size, args=(world_size, port))

    def test_llama_usp_decoder_mrope_parity(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(run_mrope_parity_case, nprocs=world_size, args=(world_size, port))

    def test_mrope_offline_non_usp_full_flow_backward(self):
        world_size = 1
        port = get_available_port()
        mp.spawn(
            run_mrope_offline_full_flow_fa_case,
            nprocs=world_size,
            args=(world_size, port),
        )

    def test_mrope_offline_usp_full_flow_backward(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(
            run_mrope_offline_full_flow_usp_case,
            nprocs=world_size,
            args=(world_size, port),
        )


if __name__ == "__main__":
    unittest.main()
