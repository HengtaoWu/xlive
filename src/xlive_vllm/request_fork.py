"""Construct vLLM child requests after the action gate resolves."""

from __future__ import annotations

from copy import deepcopy

from .kv_manager import XLiveKVCacheManager
from .protocol import BranchPlan
from .request_bridge import VllmBranchRequestFactory


class VllmChildRequestFactory:
    def __init__(
        self,
        lora_factory: VllmBranchRequestFactory,
        *,
        dialogue_max_tokens: int | None = None,
        tool_max_tokens: int | None = 512,
    ) -> None:
        self.lora_factory = lora_factory
        self.dialogue_max_tokens = dialogue_max_tokens
        self.tool_max_tokens = tool_max_tokens

    def build(self, parent, plan: BranchPlan):
        from vllm.v1.request import Request

        if parent.output_token_ids:
            raise ValueError("Shared parent must not contain generated tokens")
        if parent.pooling_params is not None:
            raise ValueError("X-Live split supports generation requests only")
        if parent.mm_features:
            raise ValueError("X-Live split v1 supports text-only requests")
        if parent.resumable:
            raise ValueError("X-Live split v1 does not support resumable requests")

        loras = self.lora_factory.lora_requests(plan)
        children = []
        for branch, lora in zip(plan.branches, loras, strict=True):
            sampling = deepcopy(parent.sampling_params)
            limit = (
                self.dialogue_max_tokens
                if branch.kind.value == "dialogue"
                else self.tool_max_tokens
            )
            if limit is not None:
                sampling.max_tokens = limit
            child = Request(
                # Preserve the external id for Dialogue so vLLM's frontend
                # output processor continues streaming the registered request.
                # Tool remains an internal sibling and is routed separately.
                request_id=(
                    parent.request_id
                    if branch.kind.value == "dialogue"
                    else plan.request_id(branch)
                ),
                prompt_token_ids=list(parent.prompt_token_ids),
                sampling_params=sampling,
                pooling_params=None,
                client_index=parent.client_index,
                arrival_time=parent.arrival_time,
                lora_request=lora,
                cache_salt=None,
                priority=parent.priority,
                trace_headers=parent.trace_headers,
                block_hasher=parent._block_hasher,
                resumable=False,
            )
            XLiveKVCacheManager.mark_tail_branch(child)
            child._xlive_parent_request_id = parent.request_id
            child._xlive_branch = branch.kind.value
            child._xlive_action_id = plan.decision.action_id
            child._xlive_expected_tool_name = branch.expected_tool_name
            children.append(child)
        return tuple(children)
