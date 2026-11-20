from collections.abc import Callable, Iterable, MutableMapping
from typing import TypeVar


RequestT = TypeVar("RequestT")
StateT = TypeVar("StateT")


def stage_candidates(
    running: Iterable[RequestT],
    waiting: Iterable[RequestT],
    skipped_waiting: Iterable[RequestT],
    *,
    stage_of: Callable[[RequestT], str],
    tail_stage: str,
) -> list[RequestT]:
    waiting_items = tuple(waiting)
    skipped_items = tuple(skipped_waiting)
    visible = list(running)
    visible.extend(
        request
        for request in (*waiting_items, *skipped_items)
        if stage_of(request) == tail_stage
    )
    if not visible:
        visible.extend(waiting_items)
        visible.extend(skipped_items)
    return visible


def register_boundary(
    cache: MutableMapping[str, StateT],
    pending: MutableMapping[str, set[str]],
    parent_id: str,
    state: StateT,
    child_ids: set[str],
) -> None:
    if parent_id in cache or parent_id in pending:
        raise RuntimeError(f"Duplicate shared boundary: {parent_id}")
    if not child_ids:
        raise ValueError("A shared boundary must have at least one child")
    cache[parent_id] = state
    pending[parent_id] = set(child_ids)


def consume_boundary_branch(
    cache: MutableMapping[str, StateT],
    pending: MutableMapping[str, set[str]],
    parent_id: str,
    child_id: str,
) -> None:
    child_ids = pending.get(parent_id)
    if child_ids is None or child_id not in child_ids:
        raise RuntimeError(f"Unexpected tail branch {child_id} for {parent_id}")
    child_ids.remove(child_id)
    if not child_ids:
        pending.pop(parent_id)
        cache.pop(parent_id, None)


def release_boundaries(
    cache: MutableMapping[str, StateT],
    pending: MutableMapping[str, set[str]],
    parent_ids: Iterable[str],
) -> None:
    for parent_id in parent_ids:
        pending.pop(parent_id, None)
        cache.pop(parent_id, None)
