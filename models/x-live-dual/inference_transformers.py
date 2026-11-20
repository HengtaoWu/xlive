"""Minimal Transformers inference for the self-contained X-Live Dual model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ACTIONS = ("NO_TOOL", "DELEGATE", "UPDATE", "INSPECT", "CANCEL")


@dataclass(frozen=True, slots=True)
class XLiveDualOutput:
    dialogue: str
    action: str
    tool_call: dict[str, Any] | None


def _parse_tool_call(text: str) -> dict[str, Any]:
    text = text.strip()
    prefix, suffix = "<tool_call>", "</tool_call>"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise RuntimeError(f"Invalid Tool branch output: {text!r}")
    payload = json.loads(text[len(prefix) : -len(suffix)])
    if (
        not isinstance(payload, dict)
        or set(payload) != {"name", "arguments"}
        or not isinstance(payload["name"], str)
        or not isinstance(payload["arguments"], dict)
    ):
        raise RuntimeError(f"Invalid Tool call: {payload!r}")
    return payload


class _DualBranches:
    def __init__(self, base: Any, dialogue_config: Any, tool_config: Any) -> None:
        import torch
        from peft import get_peft_model
        from peft.tuners.lora.layer import LoraLayer

        self.peft_model = get_peft_model(
            base, dialogue_config, adapter_name="dialogue"
        )
        self.peft_model.add_adapter("tool", tool_config)
        self.layers = self.peft_model.get_base_model().model.layers
        self.action_embedding = torch.nn.Embedding(
            len(ACTIONS), base.config.hidden_size
        )
        self._action_ids = None
        self._positions = None
        self._injected = False
        self._bypass = False
        self._lora_layers = [
            module
            for module in self.peft_model.modules()
            if isinstance(module, LoraLayer)
        ]
        if not self._lora_layers:
            raise RuntimeError("No LoRA layers were created")
        self._hook = self.layers[44].register_forward_pre_hook(
            self._inject_action, with_kwargs=True
        )

    def _inject_action(self, _module: Any, args: tuple, kwargs: dict):
        import torch

        if self._bypass or self._injected:
            return args, kwargs
        if self._action_ids is None or self._positions is None or not args:
            raise RuntimeError("Action conditioning context is missing")
        hidden = args[0]
        action = self.action_embedding(self._action_ids).to(hidden.dtype)
        conditioned = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        conditioned[rows, self._positions] += action
        self._injected = True
        return (conditioned, *args[1:]), kwargs

    def generate(
        self,
        *,
        adapter_names: list[str],
        action_ids: Any,
        positions: Any,
        **inputs: Any,
    ):
        self._action_ids = action_ids
        self._positions = positions
        self._injected = False
        try:
            return self.peft_model.generate(
                adapter_names=adapter_names,
                **inputs,
            )
        finally:
            self._action_ids = None
            self._positions = None
            self._injected = False


class XLiveDualForGeneration:
    """Reference, non-server implementation of X-Live Dual generation."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        dtype: Any,
        device_map: Any = "auto",
    ) -> None:
        import torch
        from peft import PeftConfig, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(model_dir).expanduser().resolve()
        self.root = root
        self.tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.system_prompt = (root / "system_prompt.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.tools = json.loads((root / "tools.json").read_text(encoding="utf-8"))

        base = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        dialogue_config = PeftConfig.from_pretrained(root / "dialogue")
        tool_config = PeftConfig.from_pretrained(root / "tool")
        self.branches = _DualBranches(base, dialogue_config, tool_config)

        for branch in ("dialogue", "tool"):
            state = load_file(str(root / branch / "adapter_model.safetensors"))
            result = set_peft_model_state_dict(
                self.branches.peft_model,
                state,
                adapter_name=branch,
            )
            if result.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected {branch} LoRA keys: {result.unexpected_keys[:3]}"
                )

        split_device = next(self.branches.layers[44].parameters()).device
        embedding = load_file(str(root / "action_embedding.safetensors"))[
            "action_embedding.weight"
        ]
        self.branches.action_embedding.to(device=split_device, dtype=dtype)
        with torch.no_grad():
            self.branches.action_embedding.weight.copy_(
                embedding.to(device=split_device, dtype=dtype)
            )

        gate_config = json.loads(
            (root / "action_gate/config.json").read_text(encoding="utf-8")
        )
        if gate_config["actions"] != list(ACTIONS):
            raise ValueError("Action order does not match the checkpoint")
        if gate_config["layer"] != 44:
            raise ValueError("Expected the Action Gate at layer 44")
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(gate_config["hidden_size"]),
            torch.nn.Linear(gate_config["hidden_size"], len(ACTIONS)),
        )
        self.gate.load_state_dict(
            load_file(str(root / "action_gate/model.safetensors"))
        )
        gate_device = next(self.branches.layers[43].parameters()).device
        self.gate.to(device=gate_device, dtype=torch.float32).eval()
        self.input_device = (
            self.branches.peft_model.get_input_embeddings().weight.device
        )
        self.branches.peft_model.eval()
        for parameter in self.branches.peft_model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path = ".",
        *,
        dtype: Any | None = None,
        device_map: Any = "auto",
    ) -> "XLiveDualForGeneration":
        import torch

        return cls(
            model_dir,
            dtype=dtype or torch.bfloat16,
            device_map=device_map,
        )

    def _encode(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        history = list(messages)
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": self.system_prompt})
        prompt = self.tokenizer.apply_chat_template(
            history,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return {
            key: value.to(self.input_device) for key, value in encoded.items()
        }

    def _predict_action(self, encoded: dict[str, Any]) -> int:
        import torch

        captured: list[Any] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :].detach())

        hook = self.branches.layers[43].register_forward_hook(capture)
        self.branches._bypass = True
        try:
            with torch.inference_mode(), self.branches.peft_model.disable_adapter():
                self.branches.peft_model.get_base_model().model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            self.branches._bypass = False
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Failed to capture the layer-44 hidden state")
        with torch.inference_mode():
            logits = self.gate(
                captured[0].float().to(next(self.gate.parameters()).device)
            )
        return int(logits.argmax(dim=-1).item())

    def generate(
        self,
        messages: str | list[dict[str, str]],
        *,
        dialogue_max_new_tokens: int = 192,
        tool_max_new_tokens: int = 384,
    ) -> XLiveDualOutput:
        import torch

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        encoded = self._encode(messages)
        prompt_length = encoded["input_ids"].shape[1]
        action_id = self._predict_action(encoded)
        action = ACTIONS[action_id]
        branches = ["dialogue"] if action == "NO_TOOL" else ["dialogue", "tool"]
        batch = {
            key: value.repeat(len(branches), *([1] * (value.ndim - 1)))
            for key, value in encoded.items()
        }
        action_ids = torch.full(
            (len(branches),),
            action_id,
            dtype=torch.long,
            device=self.branches.action_embedding.weight.device,
        )
        positions = torch.full(
            (len(branches),),
            prompt_length - 1,
            dtype=torch.long,
            device=self.branches.action_embedding.weight.device,
        )
        with torch.inference_mode():
            output = self.branches.generate(
                adapter_names=branches,
                action_ids=action_ids,
                positions=positions,
                **batch,
                do_sample=False,
                max_new_tokens=(
                    dialogue_max_new_tokens
                    if action == "NO_TOOL"
                    else max(dialogue_max_new_tokens, tool_max_new_tokens)
                ),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        dialogue = self.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()
        tool_call = None
        if action != "NO_TOOL":
            tool_text = self.tokenizer.decode(
                output[1, prompt_length:], skip_special_tokens=True
            ).strip()
            tool_call = _parse_tool_call(tool_text)
            if tool_call["name"].upper() != action:
                raise RuntimeError(
                    f"Action/Tool mismatch: {action} != {tool_call['name']}"
                )
        return XLiveDualOutput(dialogue, action, tool_call)
