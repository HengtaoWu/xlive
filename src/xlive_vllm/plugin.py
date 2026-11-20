"""vLLM general plugin entry point."""

from .constants import MODEL_ARCH


def register() -> None:
    from vllm import ModelRegistry

    from .kv_groups import install_kv_group_patch

    ModelRegistry.register_model(
        MODEL_ARCH,
        "xlive_vllm.model:XLiveActionGatedQwen3MoeForCausalLM",
    )
    install_kv_group_patch()
