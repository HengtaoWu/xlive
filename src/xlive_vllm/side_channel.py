"""Serializable scheduler/worker messages for staged split execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .protocol import ActionDecision


SCHEDULER_ATTR = "xlive_split_work"
RUNNER_ATTR = "xlive_gate_results"
BOUNDARY_RELEASE_ATTR = "xlive_boundary_release_ids"


class WorkStage(str, Enum):
    SHARED_PREFILL = "shared_prefill"
    TAIL_PREFILL = "tail_prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class SplitWorkItem:
    request_id: str
    stage: WorkStage
    parent_request_id: str
    branch: str | None = None
    action_id: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.parent_request_id:
            raise ValueError("request ids must not be empty")
        if self.stage is WorkStage.SHARED_PREFILL:
            if self.branch is not None or self.action_id is not None:
                raise ValueError("shared prefill must not carry branch/action state")
        else:
            if self.branch not in ("dialogue", "tool"):
                raise ValueError(f"invalid branch for {self.stage.value}: {self.branch}")
            if self.action_id is None or not 0 <= self.action_id < 5:
                raise ValueError("tail/decode work requires a valid action id")


@dataclass(frozen=True)
class GateResultMessage:
    parent_request_id: str
    action: str
    action_id: int
    logits: tuple[float, ...]

    def decision(self) -> ActionDecision:
        decision = ActionDecision(self.action, self.action_id, self.logits)
        # Also checks that the serialized id matches the logits argmax.
        predicted = ActionDecision.from_logits(self.logits)
        if predicted.action_id != decision.action_id:
            raise ValueError("serialized gate action disagrees with logits")
        return decision


def attach_split_work(scheduler_output: Any, items: tuple[SplitWorkItem, ...]) -> None:
    request_ids = set(scheduler_output.num_scheduled_tokens)
    item_ids = {item.request_id for item in items}
    if item_ids != request_ids:
        raise ValueError(
            f"split work is not aligned with scheduled requests: {item_ids} != {request_ids}"
        )
    setattr(scheduler_output, SCHEDULER_ATTR, items)


def get_split_work(scheduler_output: Any) -> tuple[SplitWorkItem, ...]:
    items = getattr(scheduler_output, SCHEDULER_ATTR, None)
    if items is None:
        raise RuntimeError("SchedulerOutput has no X-Live split work metadata")
    return tuple(items)


def attach_gate_results(model_runner_output: Any, results: tuple[GateResultMessage, ...]) -> None:
    setattr(model_runner_output, RUNNER_ATTR, results)


def pop_gate_results(model_runner_output: Any) -> tuple[GateResultMessage, ...]:
    results = tuple(getattr(model_runner_output, RUNNER_ATTR, ()))
    if hasattr(model_runner_output, RUNNER_ATTR):
        delattr(model_runner_output, RUNNER_ATTR)
    return results


def attach_boundary_releases(
    scheduler_output: Any,
    parent_request_ids: tuple[str, ...],
) -> None:
    if not parent_request_ids or any(
        not request_id for request_id in parent_request_ids
    ):
        raise ValueError("boundary release ids must not be empty")
    setattr(scheduler_output, BOUNDARY_RELEASE_ATTR, parent_request_ids)


def get_boundary_releases(scheduler_output: Any) -> tuple[str, ...]:
    return tuple(getattr(scheduler_output, BOUNDARY_RELEASE_ATTR, ()))
