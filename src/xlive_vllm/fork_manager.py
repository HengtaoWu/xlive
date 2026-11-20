"""Layer-selective block ownership operations for vLLM 0.19.1."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .constants import SPLIT_LAYER, TOTAL_LAYERS


@dataclass(frozen=True)
class ForkedBlockTables:
    request_id: str
    lower_block_copy: tuple[int, int] | None = None


class LayeredKVForkManager:
    """Manage shared and private KV blocks. Prefix caching must be disabled."""

    def __init__(self, manager: KVCacheManager) -> None:
        groups = manager.kv_cache_config.kv_cache_groups
        if len(groups) != 2:
            raise ValueError(f"Expected two X-Live KV groups, got {len(groups)}")
        if not all(type(group.kv_cache_spec) is FullAttentionSpec for group in groups):
            raise TypeError("X-Live v1 requires two exact FullAttentionSpec groups")
        if [len(group.layer_names) for group in groups] != [
            SPLIT_LAYER,
            TOTAL_LAYERS - SPLIT_LAYER,
        ]:
            raise ValueError(f"X-Live KV groups must contain exactly {SPLIT_LAYER} and {TOTAL_LAYERS - SPLIT_LAYER} layers")
        if manager.enable_caching:
            raise ValueError(
                "X-Live layer fork v1 requires enable_prefix_caching=False"
            )
        self.manager = manager
        self.lower = manager.coordinator.single_type_managers[0]
        self.tail = manager.coordinator.single_type_managers[1]
        if self.lower.block_size != self.tail.block_size:
            raise ValueError("Lower and tail block sizes must match")

    def fork_prompt(
        self,
        parent_request_id: str,
        child_request_ids: tuple[str, ...],
        num_prompt_tokens: int,
    ) -> tuple[ForkedBlockTables, ...]:
        """Alias lower prompt blocks and allocate private tail prompt blocks."""
        if not child_request_ids or len(set(child_request_ids)) != len(child_request_ids):
            raise ValueError("child_request_ids must be non-empty and unique")
        for child_id in child_request_ids:
            if child_id != parent_request_id:
                self._validate_new_request_id(child_id)
        parent_lower = self.lower.req_to_blocks.get(parent_request_id)
        if not parent_lower:
            raise KeyError(f"Parent has no shared prompt blocks: {parent_request_id}")
        if self.tail.req_to_blocks.get(parent_request_id):
            raise RuntimeError("Parent unexpectedly owns tail blocks")
        expected = cdiv(num_prompt_tokens, self.lower.block_size)
        if len(parent_lower) != expected:
            raise ValueError(
                f"Parent lower blocks do not cover prompt: {len(parent_lower)} != {expected}"
            )
        # Copy partial lower blocks per child to prevent overlapping decode writes.
        partial_final = num_prompt_tokens % self.lower.block_size != 0
        cow_needed = sum(
            child_id != parent_request_id and partial_final
            for child_id in child_request_ids
        )
        tail_needed = expected * len(child_request_ids)
        free = self.manager.block_pool.get_num_free_blocks()
        total_needed = tail_needed + cow_needed
        if total_needed > free:
            raise MemoryError(
                f"Need {tail_needed} private tail blocks and {cow_needed} "
                f"lower COW blocks, only {free} are free"
            )

        parent_lower = list(parent_lower)
        outputs = []
        for child_id in child_request_ids:
            lower_block_copy = None
            if child_id != parent_request_id:
                shared_blocks = parent_lower[:-1] if partial_final else parent_lower
                self.manager.block_pool.touch(shared_blocks)
                self.lower.req_to_blocks[child_id] = list(shared_blocks)
                if partial_final:
                    private = self.lower.allocate_new_blocks(
                        child_id, num_prompt_tokens, num_prompt_tokens
                    )
                    if len(private) != 1:
                        raise RuntimeError("Expected one lower-layer COW block")
                    lower_block_copy = (
                        parent_lower[-1].block_id,
                        private[0].block_id,
                    )
            self.tail.allocate_new_blocks(
                child_id, num_prompt_tokens, num_prompt_tokens
            )
            outputs.append(ForkedBlockTables(child_id, lower_block_copy))

        # Dialogue may reuse the parent's external request id. In that case its
        # original lower-block ownership becomes Dialogue ownership in place.
        if parent_request_id not in child_request_ids:
            self.lower.free(parent_request_id)
        return tuple(outputs)

    def _validate_new_request_id(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if self.lower.req_to_blocks.get(request_id) or self.tail.req_to_blocks.get(request_id):
            raise ValueError(f"Request already has KV blocks: {request_id}")
