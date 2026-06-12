import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "regenerate_train_data.py"
SPEC = importlib.util.spec_from_file_location("regenerate_train_data", MODULE_PATH)
regenerate_train_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regenerate_train_data)


class FakeResponseMessage:
    content = "regenerated answer"
    reasoning_content = None


class FakeChoice:
    message = FakeResponseMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = FakeChat()


def make_args(**overrides):
    args = SimpleNamespace(
        model="qwen2.5-vl",
        max_tokens=4096,
        temperature=0.8,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        reasoning="none",
        is_gpt_oss=False,
        is_vlm=True,
        image_field_names=list(regenerate_train_data.IMAGE_FIELD_NAMES),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestRegenerateTrainDataVlm(unittest.TestCase):

    def test_parse_arguments_defaults_max_tokens_to_4096(self):
        argv = [
            "regenerate_train_data.py",
            "--model",
            "qwen2.5-vl",
            "--input-file-path",
            "input.jsonl",
            "--output-file-path",
            "output.jsonl",
            "--server-address",
            "localhost:30000",
        ]
        with patch("sys.argv", argv):
            args = regenerate_train_data.parse_arguments()

        self.assertEqual(args.max_tokens, 4096)
        self.assertFalse(args.is_vlm)

    def test_get_image_urls_accepts_images_list(self):
        row = {"images": ["", "/tmp/a.png", {"url": "https://example.com/b.jpg"}]}

        self.assertEqual(
            regenerate_train_data.get_image_urls(
                row, list(regenerate_train_data.IMAGE_FIELD_NAMES)
            ),
            ["/tmp/a.png", "https://example.com/b.jpg"],
        )

    def test_injects_top_level_image_into_first_user_message(self):
        fake_client = FakeClient()
        row = {
            "image": "/tmp/chart.png",
            "conversations": [
                {"role": "user", "content": "Describe the chart."},
                {"role": "assistant", "content": "old answer"},
            ],
        }

        with patch.object(regenerate_train_data, "OpenAI", return_value=fake_client):
            result = regenerate_train_data.call_sglang(
                make_args(), "localhost:30000", row
            )

        self.assertEqual(result["status"], "success")
        messages = fake_client.chat.completions.calls[0]["messages"]
        user_content = messages[0]["content"]
        self.assertEqual(
            user_content[0],
            {"type": "image_url", "image_url": {"url": "/tmp/chart.png"}},
        )
        self.assertEqual(
            user_content[1], {"type": "text", "text": "Describe the chart."}
        )
        self.assertEqual(
            result["conversations"][-1],
            {"role": "assistant", "content": "regenerated answer"},
        )
        self.assertEqual(result["image"], "/tmp/chart.png")

    def test_existing_inline_image_is_normalized_without_duplicate_injection(self):
        fake_client = FakeClient()
        row = {
            "image": "/tmp/top-level.png",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/tmp/inline.png"},
                        {"type": "text", "text": "What is shown?"},
                    ],
                },
                {"role": "assistant", "content": "old answer"},
            ],
        }

        with patch.object(regenerate_train_data, "OpenAI", return_value=fake_client):
            result = regenerate_train_data.call_sglang(
                make_args(), "localhost:30000", row
            )

        self.assertEqual(result["status"], "success")
        messages = fake_client.chat.completions.calls[0]["messages"]
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "image_url", "image_url": {"url": "/tmp/inline.png"}},
                {"type": "text", "text": "What is shown?"},
            ],
        )

    def test_text_only_rows_still_use_string_content_when_vlm_enabled(self):
        fake_client = FakeClient()
        row = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "old answer"},
            ],
        }

        with patch.object(regenerate_train_data, "OpenAI", return_value=fake_client):
            result = regenerate_train_data.call_sglang(
                make_args(), "localhost:30000", row
            )

        self.assertEqual(result["status"], "success")
        messages = fake_client.chat.completions.calls[0]["messages"]
        self.assertEqual(messages[0], {"role": "user", "content": "Hello"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
