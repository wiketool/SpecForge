import torch
import pytest

from specforge.core.eagle3 import OnlineEagle3Model
from specforge.data.preprocessing import OfflineEagle3Dataset


def _sample(seq_len=12, hidden_size=4):
    return {
        "input_ids": torch.arange(seq_len),
        "loss_mask": torch.ones(seq_len, dtype=torch.long),
        "hidden_state": torch.randn(1, seq_len, hidden_size),
        "aux_hidden_state": torch.randn(1, seq_len, hidden_size),
        "position_ids": torch.stack(
            [
                torch.arange(seq_len),
                torch.arange(seq_len) + 100,
                torch.arange(seq_len) + 200,
            ],
            dim=0,
        ),
    }


def test_process_data_usp_slices_mrope_position_ids_seq_last():
    item = OfflineEagle3Dataset.process_data_usp(
        _sample(seq_len=12),
        max_len=12,
        ttt_length=1,
        sp_rank=1,
        sp_size=3,
        ring_rank=0,
        sp_ring_size=1,
    )

    assert item["input_ids"].shape == (1, 5)
    assert item["position_ids"].shape == (3, 1, 4)
    assert torch.equal(item["position_ids"][0, 0], torch.arange(4, 8))
    assert torch.equal(item["position_ids"][1, 0], torch.arange(104, 108))
    assert torch.equal(item["position_ids"][2, 0], torch.arange(204, 208))


def test_process_data_keeps_mrope_position_ids_for_non_usp_baseline():
    item = OfflineEagle3Dataset.process_data(_sample(seq_len=12), max_len=8)

    assert item["position_ids"].shape == (3, 1, 8)
    assert torch.equal(item["position_ids"][0, 0], torch.arange(8))
    assert torch.equal(item["position_ids"][1, 0], torch.arange(100, 108))
    assert torch.equal(item["position_ids"][2, 0], torch.arange(200, 208))


def test_prepare_position_ids_preserves_mrope_shape_for_non_usp_baseline():
    model = OnlineEagle3Model(draft_model=None, attention_backend="fa")
    position_ids = torch.arange(18).view(3, 1, 6)

    prepared = model._prepare_position_ids(
        position_ids=position_ids,
        seq_length=6,
        past_key_values_length=0,
        device=torch.device("cpu"),
        is_vlm=True,
        input_ids=torch.zeros(1, 6, dtype=torch.long),
        image_grid_thw=None,
    )

    assert prepared.shape == (3, 1, 6)
    assert torch.equal(prepared, position_ids)

    prepared_from_2d = model._prepare_position_ids(
        position_ids=position_ids[:, 0],
        seq_length=6,
        past_key_values_length=0,
        device=torch.device("cpu"),
        is_vlm=True,
        input_ids=torch.zeros(1, 6, dtype=torch.long),
        image_grid_thw=None,
    )

    assert prepared_from_2d.shape == (3, 1, 6)
    assert torch.equal(prepared_from_2d, position_ids)


def test_process_data_usp_keeps_local_linear_fallback_when_position_ids_optional():
    data = _sample(seq_len=12)
    data.pop("position_ids")

    item = OfflineEagle3Dataset.process_data_usp(
        data,
        max_len=12,
        ttt_length=1,
        sp_rank=0,
        sp_size=4,
        ring_rank=0,
        sp_ring_size=2,
    )

    assert item["input_ids"].shape == (1, 4)
    assert item["position_ids"].shape == (1, 3)
    assert torch.equal(item["position_ids"][0], torch.arange(3))


def test_process_data_usp_requires_position_ids_when_requested():
    data = _sample(seq_len=12)
    data.pop("position_ids")

    with pytest.raises(KeyError, match="position_ids are required"):
        OfflineEagle3Dataset.process_data_usp(
            data,
            max_len=12,
            ttt_length=1,
            sp_rank=0,
            sp_size=4,
            ring_rank=0,
            sp_ring_size=2,
            sample_id="0:data_0.ckpt",
            sample_path="/tmp/data_0.ckpt",
            require_position_ids=True,
        )
