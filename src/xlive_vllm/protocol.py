"""Branch metadata shared by the scheduler and model runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .constants import ACTIONS, ACTION_TO_ID


TOOL_ACTIONS = frozenset(ACTIONS[1:])


class BranchKind(str, Enum):
    DIALOGUE = "dialogue"
    TOOL = "tool"


@dataclass(frozen=True)
class ActionDecision:
    action: str
    action_id: int
    logits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in ACTION_TO_ID:
            raise ValueError(f"Unknown action: {self.action}")
        expected = ACTION_TO_ID[self.action]
        if self.action_id != expected:
            raise ValueError(
                f"Action id mismatch: {self.action} is {expected}, got {self.action_id}"
            )
        if self.logits and len(self.logits) != len(ACTIONS):
            raise ValueError(
                f"Expected {len(ACTIONS)} action logits, got {len(self.logits)}"
            )

    @classmethod
    def from_logits(cls, logits: Sequence[float]) -> "ActionDecision":
        values = tuple(float(value) for value in logits)
        if len(values) != len(ACTIONS):
            raise ValueError(
                f"Expected {len(ACTIONS)} action logits, got {len(values)}"
            )
        action_id = max(range(len(values)), key=values.__getitem__)
        return cls(ACTIONS[action_id], action_id, values)

    @property
    def needs_tool(self) -> bool:
        return self.action in TOOL_ACTIONS


@dataclass(frozen=True)
class BranchSpec:
    kind: BranchKind
    adapter_name: str
    request_suffix: str
    action_id: int
    expected_tool_name: str | None


@dataclass(frozen=True)
class BranchPlan:
    parent_request_id: str
    decision: ActionDecision
    branches: tuple[BranchSpec, ...]

    @classmethod
    def build(
        cls,
        parent_request_id: str,
        decision: ActionDecision,
        *,
        dialogue_adapter: str = "dialogue",
        tool_adapter: str = "tool",
    ) -> "BranchPlan":
        if not parent_request_id:
            raise ValueError("parent_request_id must not be empty")
        branches = [BranchSpec(
            kind=BranchKind.DIALOGUE,
            adapter_name=dialogue_adapter,
            request_suffix="dialogue",
            action_id=decision.action_id,
            expected_tool_name=None,
        )]
        if decision.needs_tool:
            branches.append(BranchSpec(
                kind=BranchKind.TOOL,
                adapter_name=tool_adapter,
                request_suffix="tool",
                action_id=decision.action_id,
                expected_tool_name=decision.action.lower(),
            ))
        return cls(parent_request_id, decision, tuple(branches))

    def request_id(self, branch: BranchSpec) -> str:
        if branch not in self.branches:
            raise ValueError("branch does not belong to this plan")
        return f"{self.parent_request_id}::{branch.request_suffix}"
