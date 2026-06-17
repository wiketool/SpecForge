import torch

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


def test_process_data_usp_keeps_linear_fallback_expanded_for_ulysses():
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
    assert item["position_ids"].shape == (1, 6)
    assert torch.equal(item["position_ids"][0], torch.arange(6))
