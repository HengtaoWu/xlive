import json
import os
from pathlib import Path

MODEL_ARCH = "XLiveActionGatedQwen3MoeForCausalLM"
TOTAL_LAYERS = 48
ACTION_CHECKPOINT_ENV = "XLIVE_ACTION_CHECKPOINT"


def checkpoint_split_layer() -> int:
    checkpoint = os.environ.get(ACTION_CHECKPOINT_ENV)
    if not checkpoint:
        return 44
    config_path = Path(checkpoint) / "action_gate" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer = config["layer"]
    if type(layer) is not int or not 0 < layer < TOTAL_LAYERS:
        raise ValueError(f"Invalid action gate layer in {config_path}: {layer!r}")
    return layer


# Read the split layer from the same checkpoint in all processes.
SPLIT_LAYER = checkpoint_split_layer()
ACTIONS = ("NO_TOOL", "DELEGATE", "UPDATE", "INSPECT", "CANCEL")
ACTION_TO_ID = {action: index for index, action in enumerate(ACTIONS)}
