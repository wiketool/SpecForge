from typing import List, Optional, Tuple, Union

import torch


ImageGridThw = Optional[
    Union[
        torch.Tensor,
        List[Optional[torch.Tensor]],
        Tuple[Optional[torch.Tensor], ...],
    ]
]


def flatten_image_grid_thw(value: ImageGridThw) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1, 3)
    if isinstance(value, (list, tuple)):
        tensors = [
            item.reshape(-1, 3)
            for item in value
            if isinstance(item, torch.Tensor) and item.numel() > 0
        ]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)
    raise TypeError(
        f"image_grid_thw must be a tensor, list, tuple, or None, got {type(value)}"
    )


def get_tp_batch_slice(
    batch_size: int,
    *,
    tp_size: int,
    tp_rank: int,
    sharded: bool = False,
) -> slice:
    if sharded:
        return slice(0, batch_size)

    chunks = torch.arange(batch_size).chunk(tp_size)
    if tp_rank >= len(chunks):
        raise ValueError(
            f"Cannot shard batch size {batch_size} across tp_size={tp_size}; "
            f"no shard exists for tp_rank={tp_rank}."
        )

    shard_indices = chunks[tp_rank]
    if shard_indices.numel() == 0:
        return slice(0, 0)
    return slice(int(shard_indices[0].item()), int(shard_indices[-1].item()) + 1)


def get_tp_data_shard(
    tensor: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    sharded: bool = False,
) -> torch.Tensor:
    if sharded:
        return tensor
    return tensor[
        get_tp_batch_slice(
            tensor.shape[0], tp_size=tp_size, tp_rank=tp_rank, sharded=False
        )
    ]


def get_tp_image_grid_thw_shard(
    image_grid_thw: ImageGridThw,
    *,
    batch_size: int,
    tp_size: int,
    tp_rank: int,
    sharded: bool = False,
) -> Optional[torch.Tensor]:
    if image_grid_thw is None:
        return None
    if sharded:
        return flatten_image_grid_thw(image_grid_thw)
    if not isinstance(image_grid_thw, (list, tuple)):
        raise TypeError(
            "VLM image_grid_thw must be a per-sample list before TP sharding, "
            f"got {type(image_grid_thw)}."
        )
    if len(image_grid_thw) != batch_size:
        raise ValueError(
            "image_grid_thw must provide one entry per target batch sample before "
            f"TP sharding. Got {len(image_grid_thw)} entries for batch size "
            f"{batch_size}."
        )

    tp_batch_slice = get_tp_batch_slice(
        batch_size, tp_size=tp_size, tp_rank=tp_rank, sharded=False
    )
    return flatten_image_grid_thw(image_grid_thw[tp_batch_slice])
