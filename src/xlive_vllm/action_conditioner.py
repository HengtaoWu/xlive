"""Frozen action gate and learned invisible action embedding."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .constants import ACTIONS, SPLIT_LAYER


class ActionConditioner(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, len(ACTIONS)),
        )
        self.embedding = torch.nn.Embedding(len(ACTIONS), hidden_size)
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        expected_hidden_size: int,
        expected_split_layer: int = SPLIT_LAYER,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ActionConditioner":
        from safetensors.torch import load_file

        root = Path(checkpoint)
        config_path = root / "action_gate" / "config.json"
        gate_path = root / "action_gate" / "model.safetensors"
        embedding_path = root / "action_embedding.safetensors"
        for path in (config_path, gate_path, embedding_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("actions") != list(ACTIONS):
            raise ValueError("Action order in checkpoint does not match runtime")
        if config.get("hidden_size") != expected_hidden_size:
            raise ValueError(
                "Action gate hidden size mismatch: "
                f"checkpoint={config.get('hidden_size')} model={expected_hidden_size}"
            )
        if config.get("layer") != expected_split_layer:
            raise ValueError(
                "Action gate split mismatch: "
                f"checkpoint={config.get('layer')} runtime={expected_split_layer}"
            )

        module = cls(expected_hidden_size)
        module.gate.load_state_dict(load_file(str(gate_path), device="cpu"), strict=True)
        embedding_state = load_file(str(embedding_path), device="cpu")
        if set(embedding_state) != {"action_embedding.weight"}:
            raise ValueError(
                f"Unexpected action embedding keys: {sorted(embedding_state)}"
            )
        module.embedding.load_state_dict(
            {"weight": embedding_state["action_embedding.weight"]}, strict=True
        )
        if device is not None or dtype is not None:
            module.to(device=device, dtype=dtype)
        module.eval()
        module.requires_grad_(False)
        return module
