"""vLLM scheduler lifecycle for shared parent -> private children.

The scheduler runs shared prefill once, forks the request after the Action
Gate, and hands the Dialogue/Tool children to vLLM's normal batching.
"""

from __future__ import annotations

import atexit
from pathlib import Path
import json
import time

from vllm.v1.core.sched.scheduler import Scheduler

from .fork_manager import LayeredKVForkManager
from .kv_manager import (
    DECODE_STAGE,
    SHARED_STAGE,
    TAIL_STAGE,
    XLiveKVCacheManager,
)
from .lifecycle import stage_candidates
from .protocol import BranchPlan
from .request_bridge import AdapterPaths, VllmBranchRequestFactory
from .request_fork import VllmChildRequestFactory
from .side_channel import (
    SplitWorkItem,
    WorkStage,
    attach_boundary_releases,
    attach_split_work,
    pop_gate_results,
)


class XLiveScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validate_runtime_config()
        # Scheduler constructed this manager moments ago and no request has
        # entered it yet. The subclass adds behavior without changing storage.
        self.kv_cache_manager.__class__ = XLiveKVCacheManager
        self.fork_manager = LayeredKVForkManager(self.kv_cache_manager)

        config = self.vllm_config.additional_config.get("xlive", {})
        checkpoint = Path(config["checkpoint"]).resolve()
        event_path = config.get("tool_event_path")
        self.tool_event_path = Path(event_path).resolve() if event_path else None
        self._tool_event_handle = None
        if self.tool_event_path is not None:
            self.tool_event_path.parent.mkdir(parents=True, exist_ok=True)
            self._tool_event_handle = self.tool_event_path.open(
                "a", encoding="utf-8", buffering=1
            )
            atexit.register(self._tool_event_handle.close)
            from transformers import AutoTokenizer

            self.tool_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True
            )
        else:
            self.tool_tokenizer = None
        self._tool_token_ids: dict[str, list[int]] = {}
        self._inflight_shared_ids: set[str] = set()
        self._cancelled_shared_ids: set[str] = set()
        self._pending_tail_children: dict[str, set[str]] = {}
        self._pending_boundary_releases: set[str] = set()
        self._pending_lower_block_copies: list[tuple[int, int]] = []
        self.child_factory = VllmChildRequestFactory(
            VllmBranchRequestFactory(AdapterPaths(
                dialogue=checkpoint / "dialogue",
                tool=checkpoint / "tool",
            )),
            dialogue_max_tokens=config.get("dialogue_max_tokens"),
            tool_max_tokens=config.get("tool_max_tokens", 512),
        )

    def _validate_runtime_config(self) -> None:
        config = self.vllm_config.additional_config.get("xlive", {})
        if not config.get("checkpoint"):
            raise ValueError("additional_config.xlive.checkpoint is required")
        if self.cache_config.enable_prefix_caching:
            raise ValueError("X-Live split v1 requires prefix caching disabled")
        if self.scheduler_config.enable_chunked_prefill:
            raise ValueError("X-Live split v1 requires chunked prefill disabled")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("X-Live split v1 does not support speculative decoding")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("X-Live split v1 does not support pipeline parallelism")
        if self.connector is not None:
            raise ValueError("X-Live split v1 does not support KV connectors")
        if self.scheduler_config.async_scheduling:
            raise ValueError("X-Live split v1 requires synchronous scheduling")
        if not self.vllm_config.model_config.enforce_eager:
            raise ValueError("X-Live split v1 requires enforce_eager=True")
        if self.lora_config is None or self.lora_config.max_loras < 2:
            raise ValueError("X-Live split v1 requires LoRA with max_loras >= 2")

    def add_request(self, request) -> None:
        if "::dialogue" in request.request_id or "::tool" in request.request_id:
            raise ValueError("External request id uses an X-Live reserved suffix")
        if request.prompt_token_ids is None or request.prompt_embeds is not None:
            raise ValueError("X-Live split v1 requires token-id text prompts")
        if request.lora_request is not None:
            raise ValueError("External X-Live parent must not select a LoRA")
        XLiveKVCacheManager.mark_shared_parent(request)
        super().add_request(request)

    def finish_requests(self, request_ids, finished_status):
        if request_ids is None:
            expanded = None
        else:
            original = (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
            expanded = set(original)
            expanded.update(
                f"{request_id}::tool"
                for request_id in original
                if not request_id.endswith("::tool")
            )
        finished = super().finish_requests(expanded, finished_status)
        for request_id, _ in finished:
            if request_id in self._inflight_shared_ids:
                self._cancelled_shared_ids.add(request_id)
                self._pending_boundary_releases.add(request_id)
            parent_id = (
                request_id.removesuffix("::tool")
                if request_id.endswith("::tool")
                else request_id
            )
            pending = self._pending_tail_children.get(parent_id)
            if pending is not None and request_id in pending:
                pending.remove(request_id)
                if not pending:
                    self._pending_tail_children.pop(parent_id, None)
                    self._pending_boundary_releases.add(parent_id)
            if request_id.endswith("::tool"):
                self._tool_token_ids.pop(request_id, None)
        return finished

    @staticmethod
    def _batch_stage(requests):
        """Choose one executable stage for the next model invocation.

        The split model has three different forward paths and a GPU batch may
        only use one of them.  Prefer work that is already running so branch
        decoding drains before a newly arrived parent prefill is admitted.
        """
        priorities = {
            TAIL_STAGE: 0,
            DECODE_STAGE: 1,
            SHARED_STAGE: 2,
        }
        stages = {XLiveKVCacheManager.stage(request) for request in requests}
        return min(stages, key=priorities.__getitem__) if stages else None

    @staticmethod
    def _remove_other_stages(queue, target_stage):
        hidden = [
            request
            for request in queue
            if XLiveKVCacheManager.stage(request) != target_stage
        ]
        if hasattr(queue, "remove_requests"):
            queue.remove_requests(hidden)
        else:
            for request in hidden:
                queue.remove(request)
        return hidden

    @staticmethod
    def _restore_queue(queue, hidden) -> None:
        for request in hidden:
            if hasattr(queue, "add_request"):
                queue.add_request(request)
            else:
                queue.append(request)

    @staticmethod
    def _stage_candidates(running, waiting, skipped_waiting):
        return stage_candidates(
            running,
            waiting,
            skipped_waiting,
            stage_of=XLiveKVCacheManager.stage,
            tail_stage=TAIL_STAGE,
        )

    def schedule(self):
        # Batch one forward stage at a time using vLLM's allocation and accounting.
        # Restore requests from other stages after scheduling.
        visible = self._stage_candidates(
            self.running,
            self.waiting,
            self.skipped_waiting,
        )
        target_stage = self._batch_stage(visible)
        if target_stage is None:
            return super().schedule()

        hidden_running = self._remove_other_stages(self.running, target_stage)
        hidden_waiting = self._remove_other_stages(self.waiting, target_stage)
        hidden_skipped = self._remove_other_stages(
            self.skipped_waiting, target_stage
        )
        original_max_running = self.max_num_running_reqs
        # Restoring already-running requests after this call must not push the
        # scheduler beyond vLLM's configured sequence limit.
        self.max_num_running_reqs = max(
            1, original_max_running - len(hidden_running)
        )
        try:
            output = super().schedule()
        finally:
            self.max_num_running_reqs = original_max_running
            self.running.extend(hidden_running)
            self._restore_queue(self.waiting, hidden_waiting)
            self._restore_queue(self.skipped_waiting, hidden_skipped)

        if self._pending_lower_block_copies:
            output.xlive_lower_block_copies = tuple(
                self._pending_lower_block_copies
            )
            self._pending_lower_block_copies.clear()
        items = []
        for request_id in output.num_scheduled_tokens:
            request = self.requests[request_id]
            stage = XLiveKVCacheManager.stage(request)
            if stage == SHARED_STAGE:
                items.append(SplitWorkItem(
                    request_id, WorkStage.SHARED_PREFILL, request_id
                ))
            else:
                parent_id = request._xlive_parent_request_id
                work_stage = (
                    WorkStage.TAIL_PREFILL if stage == TAIL_STAGE else WorkStage.DECODE
                )
                items.append(SplitWorkItem(
                    request_id=request_id,
                    stage=work_stage,
                    parent_request_id=parent_id,
                    branch=request._xlive_branch,
                    action_id=request._xlive_action_id,
                ))
        if items:
            attach_split_work(output, tuple(items))
            self._inflight_shared_ids.update(
                item.request_id for item in items
                if item.stage is WorkStage.SHARED_PREFILL
            )
        if self._pending_boundary_releases and output.num_scheduled_tokens:
            attach_boundary_releases(
                output, tuple(sorted(self._pending_boundary_releases))
            )
            self._pending_boundary_releases.clear()
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        shared_ids = {
            item.request_id
            for item in getattr(scheduler_output, "xlive_split_work", ())
            if item.stage is WorkStage.SHARED_PREFILL
        }
        for result in pop_gate_results(model_runner_output):
            # Ignore late Gate results for requests cancelled during GPU execution.
            if result.parent_request_id in self._cancelled_shared_ids:
                continue
            self._fork_from_gate(result.parent_request_id, result.decision())
        self._inflight_shared_ids.difference_update(shared_ids)
        self._cancelled_shared_ids.difference_update(shared_ids)

        tail_ids = {
            item.request_id
            for item in getattr(scheduler_output, "xlive_split_work", ())
            if item.stage is WorkStage.TAIL_PREFILL
        }
        for item in getattr(scheduler_output, "xlive_split_work", ()):
            if item.stage is not WorkStage.TAIL_PREFILL:
                continue
            pending = self._pending_tail_children.get(item.parent_request_id)
            if pending is None or item.request_id not in pending:
                raise RuntimeError(
                    f"Unexpected tail child {item.request_id} for "
                    f"{item.parent_request_id}"
                )
            pending.remove(item.request_id)
            if not pending:
                self._pending_tail_children.pop(item.parent_request_id, None)
        tool_meta = {}
        for item in getattr(scheduler_output, "xlive_split_work", ()):
            if item.branch != "tool":
                continue
            request = self.requests.get(item.request_id)
            if request is None:
                continue
            tool_meta[item.request_id] = (
                item.parent_request_id,
                item.action_id,
                request._xlive_expected_tool_name,
            )
            index = model_runner_output.req_id_to_index.get(item.request_id)
            if index is not None and model_runner_output.sampled_token_ids:
                self._tool_token_ids.setdefault(item.request_id, []).extend(
                    model_runner_output.sampled_token_ids[index]
                )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for request_id in tail_ids:
            request = self.requests.get(request_id)
            if request is not None and request.output_token_ids:
                XLiveKVCacheManager.mark_decode_branch(request)
        for request_id, (parent_id, action_id, tool_name) in tool_meta.items():
            if request_id not in self.requests:
                self._emit_tool_event(
                    request_id, parent_id, action_id, tool_name,
                    self._tool_token_ids.pop(request_id, []),
                )
        return outputs

    def _emit_tool_event(
        self, request_id, parent_request_id, action_id, expected_tool_name, token_ids
    ) -> None:
        if self.tool_event_path is None:
            return
        record = {
            "kind": "tool",
            "request_id": request_id,
            "parent_request_id": parent_request_id,
            "action_id": action_id,
            "expected_tool_name": expected_tool_name,
            "token_ids": token_ids,
            "emitted_monotonic_ns": time.monotonic_ns(),
            "emitted_unix_ns": time.time_ns(),
        }
        if self.tool_tokenizer is not None:
            record["decoded_text"] = self.tool_tokenizer.decode(
                token_ids, skip_special_tokens=False
            )
        self._write_event(record)

    def _emit_gate_event(self, parent_request_id, decision) -> None:
        if self.tool_event_path is None:
            return
        record = {
            "kind": "gate",
            "parent_request_id": parent_request_id,
            "action": decision.action,
            "action_id": decision.action_id,
            "emitted_monotonic_ns": time.monotonic_ns(),
            "emitted_unix_ns": time.time_ns(),
        }
        self._write_event(record)

    def _write_event(self, record: dict) -> None:
        if self._tool_event_handle is None:
            raise RuntimeError("Tool event output is not configured")
        self._tool_event_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _fork_from_gate(self, parent_request_id, decision) -> None:
        parent = self.requests.get(parent_request_id)
        if parent is None:
            raise KeyError(f"Gate result references unknown parent: {parent_request_id}")
        if XLiveKVCacheManager.stage(parent) != SHARED_STAGE:
            raise RuntimeError("Gate result received for a non-shared request")
        if parent.num_computed_tokens != parent.num_prompt_tokens:
            raise RuntimeError(
                "Gate result arrived before full shared prefill: "
                f"{parent.num_computed_tokens} != {parent.num_prompt_tokens}"
            )

        self._emit_gate_event(parent_request_id, decision)

        plan = BranchPlan.build(parent_request_id, decision)
        children = self.child_factory.build(parent, plan)
        child_ids = tuple(child.request_id for child in children)
        if parent_request_id in self._pending_tail_children:
            raise RuntimeError(f"Duplicate pending branches for {parent_request_id}")
        self._pending_tail_children[parent_request_id] = set(child_ids)
        forked = self.fork_manager.fork_prompt(
            parent_request_id, child_ids, parent.num_prompt_tokens
        )
        self._pending_lower_block_copies.extend(
            table.lower_block_copy
            for table in forked
            if table.lower_block_copy is not None
        )

        # Parent is an internal prefill-only request. It never emits a token and
        # must not enter the normal finished/free path because ownership has
        # already moved to the child block tables.
        if parent in self.running:
            self.running.remove(parent)
        else:
            raise RuntimeError("Shared parent is not in the running queue")
        del self.requests[parent_request_id]

        for child in children:
            super().add_request(child)
