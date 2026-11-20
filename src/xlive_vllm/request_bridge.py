"""Version-pinned conversion from an X-Live branch plan to vLLM LoRA requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .compat import require_pinned_baseline
from .protocol import BranchKind, BranchPlan


@dataclass(frozen=True)
class AdapterPaths:
    dialogue: Path
    tool: Path

    def validate(self) -> None:
        for name, path in (("dialogue", self.dialogue), ("tool", self.tool)):
            if not path.is_dir():
                raise FileNotFoundError(f"{name} adapter directory does not exist: {path}")
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"{name} adapter config is missing: {path}")


class VllmBranchRequestFactory:
    """Create LoRA requests for Dialogue and Tool branches."""

    def __init__(
        self,
        adapter_paths: AdapterPaths,
        *,
        dialogue_lora_id: int = 1,
        tool_lora_id: int = 2,
    ) -> None:
        if dialogue_lora_id < 1 or tool_lora_id < 1:
            raise ValueError("vLLM LoRA ids are one-indexed")
        if dialogue_lora_id == tool_lora_id:
            raise ValueError("Dialogue and Tool LoRA ids must differ")
        self.adapter_paths = adapter_paths
        self.ids = {
            BranchKind.DIALOGUE: dialogue_lora_id,
            BranchKind.TOOL: tool_lora_id,
        }
        require_pinned_baseline()
        self.adapter_paths.validate()
        from vllm.lora.request import LoRARequest

        self._lora_requests = {
            kind: LoRARequest(
                lora_name=f"xlive-{kind.value}",
                lora_int_id=lora_id,
                lora_path=str(
                    self.adapter_paths.dialogue
                    if kind is BranchKind.DIALOGUE
                    else self.adapter_paths.tool
                ),
            )
            for kind, lora_id in self.ids.items()
        }

    def lora_requests(self, plan: BranchPlan):
        return tuple(self._lora_requests[branch.kind] for branch in plan.branches)
