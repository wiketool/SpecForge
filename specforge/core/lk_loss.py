import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


DEFAULT_ACCEPTANCE_RATE_CHUNK_TOKENS = 2048
_ACCEPTANCE_RATE_CHUNK_TOKENS_ENV = "SPECFORGE_ACCEPTANCE_RATE_CHUNK_TOKENS"


def _resolve_acceptance_rate_chunk_tokens(chunk_tokens: Optional[int]) -> int:
    if chunk_tokens is not None:
        return max(0, int(chunk_tokens))

    env_value = os.environ.get(_ACCEPTANCE_RATE_CHUNK_TOKENS_ENV)
    if env_value is None:
        return DEFAULT_ACCEPTANCE_RATE_CHUNK_TOKENS
    try:
        return max(0, int(env_value))
    except ValueError:
        return DEFAULT_ACCEPTANCE_RATE_CHUNK_TOKENS


def expected_acceptance_rate(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute token-wise expected acceptance rates for speculative decoding."""
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            f"target_probs and draft_probs must have the same shape, "
            f"got {target_probs.shape} and {draft_probs.shape}"
        )
    return torch.minimum(target_probs, draft_probs).sum(dim=-1)


def _masked_mean(
    values_per_token: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """Compute a masked mean, with optional distributed reduction."""
    mask = position_mask.squeeze(-1)
    if mask.dtype == torch.bool:
        mask = mask.float()
    else:
        mask = mask.to(dtype=values_per_token.dtype)

    numerator = (values_per_token * mask).sum()
    denominator = mask.sum().clamp_min(eps)
    if reduce_fn is not None:
        numerator, denominator = reduce_fn(
            local_correct=numerator, local_denom=denominator
        )
        denominator = denominator.clamp_min(eps)
    return numerator / denominator


def _acceptance_rate_per_token_from_logits(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Return per-token expected acceptance from draft logits and target probs."""
    draft_p = F.softmax(logits.to(torch.float32), dim=-1).to(target_probs.dtype)
    return expected_acceptance_rate(target_probs=target_probs, draft_probs=draft_p)


def _masked_sum(
    values_per_token: torch.Tensor,
    position_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = position_mask.squeeze(-1)
    if mask.dtype == torch.bool:
        mask = mask.float()
    else:
        mask = mask.to(dtype=values_per_token.dtype)

    numerator = (values_per_token * mask).sum()
    denominator = mask.sum()
    return numerator, denominator


def _masked_mean_from_sums(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    denominator = denominator.clamp_min(eps)
    if reduce_fn is not None:
        numerator, denominator = reduce_fn(
            local_correct=numerator, local_denom=denominator
        )
        denominator = denominator.clamp_min(eps)
    return numerator / denominator


def _compute_acceptance_rate_full(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    acceptance_rate_per_token = _acceptance_rate_per_token_from_logits(
        logits=logits,
        target_probs=target_probs,
    )
    acceptance_rate = _masked_mean(
        values_per_token=acceptance_rate_per_token,
        position_mask=position_mask,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    log_acceptance_rate_per_token = torch.where(
        acceptance_rate_per_token > 0, torch.log(acceptance_rate_per_token), 0
    )
    log_acceptance_rate = _masked_mean(
        values_per_token=log_acceptance_rate_per_token,
        position_mask=position_mask,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    return acceptance_rate, log_acceptance_rate


def _compute_acceptance_rate_chunked(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]],
    chunk_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    acceptance_numerator = logits.new_zeros((), dtype=torch.float32)
    log_acceptance_numerator = logits.new_zeros((), dtype=torch.float32)
    denominator = logits.new_zeros((), dtype=torch.float32)

    for start in range(0, logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, logits.shape[1])
        acceptance_rate_per_token = _acceptance_rate_per_token_from_logits(
            logits=logits[:, start:end, :],
            target_probs=target_probs[:, start:end, :],
        )
        if position_mask.dim() >= 3:
            mask = position_mask[:, start:end, ...]
        else:
            mask = position_mask[:, start:end]

        chunk_numerator, chunk_denominator = _masked_sum(
            values_per_token=acceptance_rate_per_token,
            position_mask=mask,
        )
        acceptance_numerator = acceptance_numerator + chunk_numerator.float()
        denominator = denominator + chunk_denominator.float()

        log_acceptance_rate_per_token = torch.where(
            acceptance_rate_per_token > 0,
            torch.log(acceptance_rate_per_token),
            torch.zeros_like(acceptance_rate_per_token),
        )
        chunk_log_numerator, _ = _masked_sum(
            values_per_token=log_acceptance_rate_per_token,
            position_mask=mask,
        )
        log_acceptance_numerator = (
            log_acceptance_numerator + chunk_log_numerator.float()
        )

    acceptance_rate = _masked_mean_from_sums(
        numerator=acceptance_numerator,
        denominator=denominator,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    log_acceptance_rate = _masked_mean_from_sums(
        numerator=log_acceptance_numerator,
        denominator=denominator,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    return acceptance_rate, log_acceptance_rate


def compute_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float = 1e-8,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
    chunk_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return masked means of acceptance and log-acceptance over valid positions."""
    resolved_chunk_tokens = _resolve_acceptance_rate_chunk_tokens(chunk_tokens)
    if resolved_chunk_tokens <= 0 or logits.shape[1] <= resolved_chunk_tokens:
        return _compute_acceptance_rate_full(
            logits=logits,
            target_probs=target_probs,
            position_mask=position_mask,
            eps=eps,
            reduce_fn=reduce_fn,
        )

    return _compute_acceptance_rate_chunked(
        logits=logits,
        target_probs=target_probs,
        position_mask=position_mask,
        eps=eps,
        reduce_fn=reduce_fn,
        chunk_tokens=resolved_chunk_tokens,
    )


def compute_lk_loss(
    *,
    kl_loss: torch.Tensor,
    acceptance_rate: torch.Tensor,
    log_acceptance_rate: torch.Tensor,
    lk_loss_type: str,
    kl_scale: float,
    kl_decay: float,
) -> torch.Tensor:
    """Compute LK loss from KL loss and acceptance rate."""
    if lk_loss_type == "alpha":
        return -log_acceptance_rate
    if lk_loss_type == "lambda":
        acc_det = acceptance_rate.detach()
        kl_weight = kl_scale * torch.exp(-kl_decay * acc_det)
        return kl_weight * kl_loss + (1 - kl_weight) * (1 - acceptance_rate)
    raise ValueError(f"Unknown lk loss type: {lk_loss_type}")
