"""Version-pinned KV group split for lower and tail transformer layers."""

from __future__ import annotations

import re
from typing import Mapping

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, KVCacheSpec

from .constants import MODEL_ARCH, SPLIT_LAYER, TOTAL_LAYERS


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _layer_index(layer_name: str) -> int:
    match = _LAYER_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"Cannot extract transformer layer from {layer_name!r}")
    return int(match.group(1))


def split_action_gated_groups(
    architectures: tuple[str, ...] | list[str],
    kv_cache_specs: Mapping[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Return checkpoint-based KV groups, or None for other models."""
    if MODEL_ARCH not in architectures:
        return None
    if len(kv_cache_specs) != TOTAL_LAYERS:
        raise ValueError(
            f"X-Live expects one KV spec per decoder layer, got {len(kv_cache_specs)}"
        )
    indexed = sorted(
        ((_layer_index(name), name, spec) for name, spec in kv_cache_specs.items()),
        key=lambda item: item[0],
    )
    if [item[0] for item in indexed] != list(range(TOTAL_LAYERS)):
        raise ValueError("X-Live KV layer indices must be exactly 0..47")
    if not all(isinstance(item[2], FullAttentionSpec) for item in indexed):
        raise TypeError("X-Live v1 split currently supports full attention only")

    lower = indexed[:SPLIT_LAYER]
    tail = indexed[SPLIT_LAYER:]
    base_lower = lower[0][2]
    base_tail = tail[0][2]
    # Keep shared and branch KV groups separate despite identical attention specs.
    # NoPrefixCacheCoordinator supports this layout with prefix caching disabled.
    if any(item[2] != base_lower for item in indexed):
        raise ValueError("X-Live Qwen3 KV specs unexpectedly differ across layers")
    return [
        KVCacheGroupSpec(
            layer_names=[item[1] for item in lower],
            kv_cache_spec=base_lower,
        ),
        KVCacheGroupSpec(
            layer_names=[item[1] for item in tail],
            kv_cache_spec=base_tail,
        ),
    ]


def install_kv_group_patch() -> None:
    """Patch only the group-selection hook, guarded by the custom architecture."""
    from vllm.v1.core import kv_cache_utils

    current = kv_cache_utils.get_kv_cache_groups
    if getattr(current, "_xlive_action_gated", False):
        return

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        architectures = tuple(vllm_config.model_config.architectures or ())
        groups = split_action_gated_groups(architectures, kv_cache_spec)
        if groups is not None:
            return groups
        return current(vllm_config, kv_cache_spec)

    get_kv_cache_groups._xlive_action_gated = True
    get_kv_cache_groups._xlive_original = current
    kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
