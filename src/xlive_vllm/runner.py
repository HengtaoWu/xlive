"""vLLM 0.19.1 GPU runner for X-Live staged dual-branch execution."""

from __future__ import annotations

import os

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from .constants import ACTION_CHECKPOINT_ENV, SPLIT_LAYER
from .side_channel import (
    GateResultMessage,
    WorkStage,
    attach_gate_results,
    get_boundary_releases,
    get_split_work,
)


class XLiveGPUModelRunner(GPUModelRunner):
    """Reuse stock execution while suppressing sampling for gate-only prefill."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._xlive_stage: WorkStage | None = None
        self._xlive_shared_request_ids: tuple[str, ...] = ()

    def load_model(self, *args, **kwargs) -> None:
        # Keep the external gate out of vLLM's base-checkpoint missing-weight
        # validation. It is a separately trained artifact, loaded only after
        # the base model and LoRA-capable wrapper are fully initialized.
        super().load_model(*args, **kwargs)
        checkpoint = os.environ.get(ACTION_CHECKPOINT_ENV)
        if not checkpoint:
            raise RuntimeError(f"Set {ACTION_CHECKPOINT_ENV} before loading X-Live")
        if not hasattr(self.model, "load_action_conditioner"):
            raise TypeError("Loaded model does not expose the X-Live action conditioner")
        self.model.load_action_conditioner(checkpoint)

    def _copy_lower_prompt_blocks(self, scheduler_output) -> None:
        """Materialize partial-block COW before a branch tail prefill.

        vLLM 0.19.1 performs request-state updates directly inside
        ``GPUModelRunner.execute_model``; it no longer calls the historical
        ``_update_states`` hook.  The scheduler has already allocated the
        destination blocks by the time this method runs, and all operations
        are enqueued on the same CUDA stream as the preceding parent prefill
        and the following branch execution.
        """
        copies = getattr(scheduler_output, "xlive_lower_block_copies", ())
        if not copies:
            return
        if len(self.kv_caches) < SPLIT_LAYER:
            raise RuntimeError(
                f"Expected at least {SPLIT_LAYER} KV tensors, got "
                f"{len(self.kv_caches)}"
            )
        max_block_id = max(max(source, destination) for source, destination in copies)
        for layer_index, cache in enumerate(self.kv_caches[:SPLIT_LAYER]):
            # FlashAttention exposes logical shape
            # [K/V, physical_block, block_token, kv_head, head_dim].
            if cache.ndim < 2 or max_block_id >= cache.shape[1]:
                raise RuntimeError(
                    f"Invalid KV COW block for layer {layer_index}: "
                    f"shape={tuple(cache.shape)} max_block_id={max_block_id}"
                )
            for source, destination in copies:
                cache[:, destination].copy_(cache[:, source])

    def execute_model(self, scheduler_output, intermediate_tensors=None, **kwargs):
        boundary_release_ids = get_boundary_releases(scheduler_output)
        if boundary_release_ids:
            self.model.release_split_boundaries(boundary_release_ids)
        # Use the full model for vLLM profiling and warmup batches.
        if not hasattr(scheduler_output, "xlive_split_work"):
            self._xlive_stage = None
            self._xlive_shared_request_ids = ()
            return super().execute_model(
                scheduler_output, intermediate_tensors, **kwargs
            )
        # Copy partial prompt blocks before attention metadata and tail prefill.
        # In vLLM 0.19.1 this replaces the removed _update_states hook.
        self._copy_lower_prompt_blocks(scheduler_output)
        items = get_split_work(scheduler_output)
        stages = {item.stage for item in items}
        if len(stages) != 1:
            raise RuntimeError(
                "X-Live split v1 cannot execute mixed shared/tail/decode batches"
            )
        stage = next(iter(stages))
        self._xlive_stage = stage
        return super().execute_model(
            scheduler_output, intermediate_tensors, **kwargs
        )

    def prepare_inputs(self, scheduler_output, batch_desc):
        batch = super().prepare_inputs(scheduler_output, batch_desc)
        if not hasattr(scheduler_output, "xlive_split_work"):
            return batch
        items = get_split_work(scheduler_output)
        by_id = {item.request_id: item for item in items}
        if len(by_id) != len(items) or set(by_id) != set(batch.req_ids):
            raise RuntimeError("X-Live work does not match the packed input batch")
        # Match branch metadata to vLLM's token-count ordering.
        ordered = tuple(by_id[request_id] for request_id in batch.req_ids)
        self.model.set_split_work(
            ordered, tuple(int(count) for count in batch.num_scheduled_tokens)
        )
        self._xlive_shared_request_ids = (
            tuple(batch.req_ids)
            if self._xlive_stage is WorkStage.SHARED_PREFILL else ()
        )
        return batch

    def sample_tokens(self, grammar_output):
        if self._xlive_stage is not WorkStage.SHARED_PREFILL:
            self._xlive_stage = None
            return super().sample_tokens(grammar_output)

        if self.execute_model_state is None:
            raise RuntimeError("Shared prefill completed without execute_model state")
        state = self.execute_model_state
        self.execute_model_state = None
        side_channel = self.model.take_gate_side_channel()
        decisions = side_channel.decisions()
        if side_channel.request_ids != self._xlive_shared_request_ids:
            raise RuntimeError("Gate result order differs from shared request order")

        output = ModelRunnerOutput(
            req_ids=list(side_channel.request_ids),
            req_id_to_index={
                request_id: index
                for index, request_id in enumerate(side_channel.request_ids)
            },
            sampled_token_ids=[[] for _ in side_channel.request_ids],
            kv_connector_output=state.kv_connector_output,
        )
        attach_gate_results(
            output,
            tuple(
                GateResultMessage(
                    parent_request_id=request_id,
                    action=decision.action,
                    action_id=decision.action_id,
                    logits=decision.logits,
                )
                for request_id, decision in zip(
                    side_channel.request_ids, decisions, strict=True
                )
            ),
        )

        for request_id in self._xlive_shared_request_ids:
            if not self._remove_request(request_id):
                raise RuntimeError(f"Missing worker parent request: {request_id}")
        self._xlive_shared_request_ids = ()
        self._xlive_stage = None
        return output
