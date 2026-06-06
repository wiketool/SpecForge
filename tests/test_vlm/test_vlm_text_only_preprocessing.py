import torch

from specforge.data.preprocessing import preprocess_vlm_conversations
from specforge.data.template import TEMPLATE_REGISTRY


class FakeEncoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeTokenizer:
    def __init__(self):
        self.last_text = ""

    def decode(self, input_ids, skip_special_tokens=False):
        return self.last_text


class FakeImageProcessor:
    patch_size = 14
    temporal_patch_size = 2
    num_channels = 3


class FakeVlmProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text_items = [
                    item["text"] for item in content if item.get("type") == "text"
                ]
                content = "\n".join(text_items)
            parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        self.tokenizer.last_text = "".join(parts)
        return self.tokenizer.last_text

    def __call__(self, **kwargs):
        assert "images" not in kwargs
        text = kwargs["text"][0]
        input_ids = torch.arange(len(text), dtype=torch.long).unsqueeze(0)
        offsets = torch.tensor(
            [[idx, idx + 1] for idx in range(len(text))], dtype=torch.long
        ).unsqueeze(0)
        return FakeEncoding(input_ids=input_ids, offset_mapping=offsets)


def test_preprocess_vlm_conversations_allows_missing_image_column():
    conversations = [
        {"role": "user", "content": "what is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    examples = {"conversations": [conversations]}

    processed = preprocess_vlm_conversations(
        FakeVlmProcessor(),
        examples,
        TEMPLATE_REGISTRY.get("qwen2-vl"),
        max_length=4096,
    )

    assert len(processed["input_ids"]) == 1
    assert processed["loss_mask"][0].sum() > 0
    assert processed["pixel_values"][0].shape == (0, 1176)
    assert processed["image_grid_thw"][0].shape == (0, 3)
