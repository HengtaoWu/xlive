"""Tensor contract at the shared/tail split."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import ActionDecision


@dataclass(frozen=True)
class PromptBoundary:
    request_ids: tuple[str, ...]
    last_token_indices: torch.Tensor

    def validate(self, packed_tokens: int) -> None:
        if self.last_token_indices.ndim != 1:
            raise ValueError("last_token_indices must be one-dimensional")
        if len(self.request_ids) != self.last_token_indices.numel():
            raise ValueError("request_ids and last_token_indices are not aligned")
        if self.last_token_indices.numel() == 0:
            raise ValueError("prompt boundary must contain at least one request")
        indices = self.last_token_indices.to(dtype=torch.long, device="cpu")
        if int(indices.min()) < 0 or int(indices.max()) >= packed_tokens:
            raise IndexError("last prompt token index is outside the packed tensor")
        if len(set(indices.tolist())) != indices.numel():
            raise ValueError("last prompt token indices must be unique")


@dataclass(frozen=True)
class GateSideChannel:
    request_ids: tuple[str, ...]
    action_ids: torch.Tensor
    logits: torch.Tensor

    def decisions(self) -> tuple[ActionDecision, ...]:
        ids = self.action_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        values = self.logits.detach().to(device="cpu", dtype=torch.float32).tolist()
        if len(ids) != len(self.request_ids) or len(values) != len(self.request_ids):
            raise ValueError("gate side-channel batch is not aligned")
        decisions = tuple(ActionDecision.from_logits(row) for row in values)
        if [decision.action_id for decision in decisions] != ids:
            raise ValueError("action_ids disagree with action logits")
        return decisions


def materialize_residual_stream(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Reconstruct the HF layer-boundary hidden state from vLLM's fused form.

    Qwen3MoeDecoderLayer keeps the residual accumulator separate and combines
    it in the following layer's fused RMSNorm. The Transformers checkpoint used
    to train our gate exposes their sum at the same boundary.
    """
    if residual is None:
        return hidden_states
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual must have the same shape")
    return hidden_states + residual


def run_gate(
    hidden_44: torch.Tensor,
    boundary: PromptBoundary,
    action_head,
    residual_44: torch.Tensor | None = None,
) -> GateSideChannel:
    """Run the gate on final prompt rows of a packed hidden-state tensor."""
    if hidden_44.ndim != 2:
        raise ValueError("hidden_44 must have shape [packed_tokens, hidden_size]")
    boundary.validate(hidden_44.shape[0])
    residual_stream = materialize_residual_stream(hidden_44, residual_44)
    indices = boundary.last_token_indices.to(hidden_44.device, dtype=torch.long)
    logits = action_head(residual_stream.index_select(0, indices))
    if logits.ndim != 2 or logits.shape != (len(boundary.request_ids), 5):
        raise ValueError(
            "action_head must return [num_requests, 5], "
            f"got {tuple(logits.shape)}"
        )
    return GateSideChannel(boundary.request_ids, logits.argmax(dim=-1), logits)


def inject_action_at_prompt_boundary(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    boundary: PromptBoundary,
    action_ids: torch.Tensor,
    action_embedding,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inject an action only at each request's final prompt-token boundary."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [packed_tokens, hidden_size]")
    boundary.validate(hidden_states.shape[0])
    if action_ids.ndim != 1 or action_ids.numel() != len(boundary.request_ids):
        raise ValueError("action_ids must have one value per request")
    indices = boundary.last_token_indices.to(hidden_states.device, dtype=torch.long)
    action_ids = action_ids.to(hidden_states.device, dtype=torch.long)
    additions = action_embedding(action_ids).to(dtype=hidden_states.dtype)
    if additions.shape != (len(boundary.request_ids), hidden_states.shape[-1]):
        raise ValueError("action embedding hidden size does not match the backbone")
    if residual is None:
        conditioned_hidden = hidden_states.clone()
        conditioned_hidden.index_add_(0, indices, additions)
        return conditioned_hidden, None
    if residual.shape != hidden_states.shape:
        raise ValueError("hidden_states and residual must have the same shape")
    conditioned_residual = residual.clone()
    conditioned_residual.index_add_(0, indices, additions)
    return hidden_states, conditioned_residual
