"""Version-pinned split Qwen3-MoE model surface for vLLM 0.19.1.

Stock vLLM still calls ``forward``. The custom model runner will call
``forward_shared`` once and ``forward_tail`` once per selected branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from itertools import islice

import torch

from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .action_conditioner import ActionConditioner
from .constants import ACTION_CHECKPOINT_ENV, SPLIT_LAYER, TOTAL_LAYERS
from .lifecycle import (
    consume_boundary_branch,
    register_boundary,
    release_boundaries,
)
from .runner_contract import (
    GateSideChannel,
    PromptBoundary,
    inject_action_at_prompt_boundary,
    materialize_residual_stream,
    run_gate,
)
from .side_channel import SplitWorkItem, WorkStage


@dataclass(frozen=True)
class SplitBoundaryState:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None


class XLiveActionGatedQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3-MoE with explicit shared/tail execution entry points."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.config.num_hidden_layers != TOTAL_LAYERS:
            raise ValueError(
                f"X-Live split expects 48 layers, got {self.config.num_hidden_layers}"
            )
        self.action_conditioner: ActionConditioner | None = None
        self._split_work: tuple[SplitWorkItem, ...] | None = None
        self._split_token_counts: tuple[int, ...] | None = None
        self._boundary_cache: dict[str, SplitBoundaryState] = {}
        self._pending_tail_request_ids: dict[str, set[str]] = {}
        self._gate_side_channel: GateSideChannel | None = None
        self._moe_layer_names = [
            layer.mlp.experts.layer_name for layer in self.model.layers
        ]

    @contextmanager
    def _moe_layer_range(self, start: int, end: int):
        context = get_forward_context()
        if context.all_moe_layers is None:
            yield
            return
        names = self._moe_layer_names[start:end]
        if context.all_moe_layers[start:end] != names:
            raise RuntimeError("X-Live MoE layer registration does not match execution order")
        previous_index = context.moe_layer_index
        # Start each segment at its expert layer; vLLM contexts default to zero.
        context.moe_layer_index = start
        try:
            yield
            if context.moe_layer_index != end:
                raise RuntimeError(
                    f"X-Live MoE execution ended at {context.moe_layer_index}, expected {end}"
                )
        finally:
            context.moe_layer_index = previous_index

    def _require_single_pipeline_stage(self) -> None:
        if self.model.start_layer != 0 or self.model.end_layer != TOTAL_LAYERS:
            raise RuntimeError(
                "The first split runner supports TP but not pipeline parallelism; "
                f"local layer range is [{self.model.start_layer}, {self.model.end_layer})"
            )

    def load_action_conditioner(self, checkpoint: str) -> None:
        parameter = next(self.parameters())
        self.action_conditioner = ActionConditioner.from_checkpoint(
            checkpoint,
            expected_hidden_size=self.config.hidden_size,
            expected_split_layer=SPLIT_LAYER,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def forward_shared(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> SplitBoundaryState:
        """Run embeddings and shared layers before the split."""
        self._require_single_pipeline_stage()
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise RuntimeError("Split runner currently requires pipeline_parallel_size=1")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.model.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        residual = None
        with self._moe_layer_range(0, SPLIT_LAYER):
            for layer in islice(self.model.layers, 0, SPLIT_LAYER):
                hidden_states, residual = layer(positions, hidden_states, residual)
        return SplitBoundaryState(hidden_states, residual)

    def decide_action(
        self,
        state: SplitBoundaryState,
        boundary: PromptBoundary,
    ) -> GateSideChannel:
        if self.action_conditioner is None:
            raise RuntimeError(
                f"Action conditioner is not loaded; set {ACTION_CHECKPOINT_ENV}"
            )
        return run_gate(
            state.hidden_states,
            boundary,
            self.action_conditioner.gate,
            state.residual,
        )

    def condition_action(
        self,
        state: SplitBoundaryState,
        boundary: PromptBoundary,
        action_ids: torch.Tensor,
    ) -> SplitBoundaryState:
        if self.action_conditioner is None:
            raise RuntimeError(
                f"Action conditioner is not loaded; set {ACTION_CHECKPOINT_ENV}"
            )
        hidden_states, residual = inject_action_at_prompt_boundary(
            state.hidden_states,
            state.residual,
            boundary,
            action_ids,
            self.action_conditioner.embedding,
        )
        return SplitBoundaryState(hidden_states, residual)

    def forward_tail(
        self,
        state: SplitBoundaryState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run branch layers and final RMSNorm with per-row LoRA adapters."""
        self._require_single_pipeline_stage()
        hidden_states, residual = state.hidden_states, state.residual
        with self._moe_layer_range(SPLIT_LAYER, TOTAL_LAYERS):
            for layer in islice(self.model.layers, SPLIT_LAYER, TOTAL_LAYERS):
                hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def set_split_work(
        self,
        items: tuple[SplitWorkItem, ...],
        token_counts: tuple[int, ...],
    ) -> None:
        if len(items) != len(token_counts) or not items:
            raise ValueError("split work and token counts must be non-empty and aligned")
        if any(count <= 0 for count in token_counts):
            raise ValueError("split work token counts must be positive")
        stages = {item.stage for item in items}
        if len(stages) != 1:
            raise RuntimeError(
                "X-Live split v1 requires a homogeneous shared/tail/decode batch"
            )
        self._split_work = items
        self._split_token_counts = token_counts

    def take_gate_side_channel(self) -> GateSideChannel:
        side_channel = self._gate_side_channel
        if side_channel is None:
            raise RuntimeError("No gate result is available for the current batch")
        self._gate_side_channel = None
        return side_channel

    @staticmethod
    def _packed_boundary(
        request_ids: tuple[str, ...],
        token_counts: tuple[int, ...],
        device: torch.device,
    ) -> PromptBoundary:
        last_indices = []
        offset = 0
        for count in token_counts:
            offset += count
            last_indices.append(offset - 1)
        return PromptBoundary(
            request_ids,
            torch.tensor(last_indices, dtype=torch.long, device=device),
        )

    def _forward_split_shared(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        items: tuple[SplitWorkItem, ...],
        token_counts: tuple[int, ...],
    ) -> torch.Tensor:
        state = self.forward_shared(
            input_ids, positions, inputs_embeds=inputs_embeds
        )
        request_ids = tuple(item.request_id for item in items)
        boundary = self._packed_boundary(request_ids, token_counts, positions.device)
        self._gate_side_channel = self.decide_action(state, boundary)
        decisions = self._gate_side_channel.decisions()

        offset = 0
        for item, count, decision in zip(
            items, token_counts, decisions, strict=True
        ):
            end = offset + count
            residual = None if state.residual is None else state.residual[offset:end]
            pending = {item.request_id}
            if decision.needs_tool:
                pending.add(f"{item.request_id}::tool")
            register_boundary(
                self._boundary_cache,
                self._pending_tail_request_ids,
                item.request_id,
                SplitBoundaryState(state.hidden_states[offset:end], residual),
                pending,
            )
            offset = end

        return materialize_residual_stream(state.hidden_states, state.residual)

    def _forward_split_tail(
        self,
        positions: torch.Tensor,
        items: tuple[SplitWorkItem, ...],
        token_counts: tuple[int, ...],
    ) -> torch.Tensor:
        hidden_parts = []
        residual_parts = []
        has_residual: bool | None = None
        for item, count in zip(items, token_counts, strict=True):
            state = self._boundary_cache.get(item.parent_request_id)
            if state is None:
                raise KeyError(
                    f"Missing layer-{SPLIT_LAYER} boundary for {item.parent_request_id}"
                )
            if state.hidden_states.shape[0] != count:
                raise RuntimeError(
                    "Tail prompt length differs from cached shared prompt: "
                    f"{count} != {state.hidden_states.shape[0]}"
                )
            current_has_residual = state.residual is not None
            if has_residual is None:
                has_residual = current_has_residual
            elif has_residual != current_has_residual:
                raise RuntimeError("Inconsistent residual representation in tail batch")
            hidden_parts.append(state.hidden_states)
            if state.residual is not None:
                residual_parts.append(state.residual)

        state = SplitBoundaryState(
            torch.cat(hidden_parts, dim=0),
            torch.cat(residual_parts, dim=0) if has_residual else None,
        )
        boundary = self._packed_boundary(
            tuple(item.request_id for item in items), token_counts, positions.device
        )
        action_ids = torch.tensor(
            [item.action_id for item in items], dtype=torch.long, device=positions.device
        )
        conditioned = self.condition_action(state, boundary, action_ids)
        output = self.forward_tail(conditioned, positions)

        for item in items:
            consume_boundary_branch(
                self._boundary_cache,
                self._pending_tail_request_ids,
                item.parent_request_id,
                item.request_id,
            )
        return output

    def release_split_boundaries(
        self,
        parent_request_ids: tuple[str, ...],
    ) -> None:
        release_boundaries(
            self._boundary_cache,
            self._pending_tail_request_ids,
            parent_request_ids,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        items = self._split_work
        token_counts = self._split_token_counts
        self._split_work = None
        self._split_token_counts = None
        if items is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if intermediate_tensors is not None:
            raise RuntimeError("X-Live split v1 does not support pipeline parallelism")
        assert token_counts is not None
        stage = items[0].stage
        if stage is WorkStage.SHARED_PREFILL:
            return self._forward_split_shared(
                input_ids, positions, inputs_embeds, items, token_counts
            )
        if stage is WorkStage.TAIL_PREFILL:
            return self._forward_split_tail(positions, items, token_counts)
        if stage is WorkStage.DECODE:
            return super().forward(input_ids, positions, None, inputs_embeds)
        raise AssertionError(f"Unhandled X-Live work stage: {stage}")
