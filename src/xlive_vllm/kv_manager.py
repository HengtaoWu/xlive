"""KVCacheManager extension for lower-only parent prefill."""

from __future__ import annotations

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager


STAGE_ATTR = "_xlive_stage"
SHARED_STAGE = "shared"
TAIL_STAGE = "tail"
DECODE_STAGE = "decode"


class XLiveKVCacheManager(KVCacheManager):
    """Allocate only group 0 while a temporary parent is in shared prefill."""

    @staticmethod
    def mark_shared_parent(request) -> None:
        setattr(request, STAGE_ATTR, SHARED_STAGE)

    @staticmethod
    def mark_tail_branch(request) -> None:
        setattr(request, STAGE_ATTR, TAIL_STAGE)

    @staticmethod
    def mark_decode_branch(request) -> None:
        setattr(request, STAGE_ATTR, DECODE_STAGE)

    @staticmethod
    def stage(request) -> str:
        return getattr(request, STAGE_ATTR, DECODE_STAGE)

    def can_fit_full_sequence(
        self,
        request,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_external_computed_tokens=0,
        num_encoder_tokens=0,
    ) -> bool:
        if self.stage(request) != SHARED_STAGE:
            return super().can_fit_full_sequence(
                request,
                num_new_computed_tokens,
                new_computed_blocks,
                num_external_computed_tokens,
                num_encoder_tokens,
            )
        if any((num_new_computed_tokens, num_external_computed_tokens, num_encoder_tokens)):
            raise RuntimeError("Shared parent does not support cached/external/encoder tokens")
        required = -(-request.num_tokens // self.coordinator.single_type_managers[0].block_size)
        current = len(
            self.coordinator.single_type_managers[0].req_to_blocks.get(
                request.request_id, ()
            )
        )
        return required - current <= self.block_pool.get_num_free_blocks()

    def allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
    ):
        if self.stage(request) != SHARED_STAGE:
            return super().allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens,
                num_external_computed_tokens,
                delay_cache_blocks,
                num_encoder_tokens,
            )
        if new_computed_blocks is not None and any(new_computed_blocks.blocks):
            raise RuntimeError("Shared parent cannot consume prefix-cached blocks")
        if any((
            num_new_computed_tokens,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
        )):
            raise RuntimeError("Unsupported feature in X-Live shared prefill")
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")

        lower = self.coordinator.single_type_managers[0]
        total_tokens = request.num_computed_tokens + num_new_tokens
        required = -(-total_tokens // lower.block_size)
        current = len(lower.req_to_blocks.get(request.request_id, ()))
        needed = max(required - current, 0)
        if needed > self.block_pool.get_num_free_blocks():
            return None
        new_lower = lower.allocate_new_blocks(
            request.request_id, total_tokens, total_tokens
        )
        # Group 1 intentionally receives no block table entries until fork.
        return KVCacheBlocks((new_lower, ()))
