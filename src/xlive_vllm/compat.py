"""Check compatibility with the pinned vLLM version."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect


PINNED_VLLM_VERSION = "0.19.1"


@dataclass(frozen=True)
class CompatibilityReport:
    installed_version: str | None
    version_matches: bool
    scheduler_hooks_present: bool
    kv_hooks_present: bool
    native_layer_fork: bool
    details: tuple[str, ...]

    @property
    def baseline_ready(self) -> bool:
        return self.version_matches and self.scheduler_hooks_present and self.kv_hooks_present

def inspect_vllm() -> CompatibilityReport:
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        return CompatibilityReport(
            None, False, False, False, False, ("vllm is not installed",)
        )

    details: list[str] = []
    try:
        scheduler_cls = import_module("vllm.v1.core.sched.scheduler").Scheduler
        scheduler_hooks = all(
            hasattr(scheduler_cls, name)
            for name in ("schedule", "add_request", "finish_requests")
        )
    except (ImportError, AttributeError) as exc:
        scheduler_hooks = False
        details.append(f"scheduler import failed: {exc}")

    try:
        manager_cls = import_module("vllm.v1.core.kv_cache_manager").KVCacheManager
        kv_hooks = all(
            hasattr(manager_cls, name) for name in ("allocate_slots", "free")
        )
        signature = inspect.signature(manager_cls.allocate_slots)
        kv_hooks = kv_hooks and "new_computed_blocks" in signature.parameters
    except (ImportError, AttributeError) as exc:
        kv_hooks = False
        details.append(f"KV manager import failed: {exc}")

    # vLLM 0.19.1 has request/cache-group block tables, but no public API for
    # sharing only a subset of transformer layers between child requests.
    native_layer_fork = False
    details.append(
        "layer-selective request fork is not a public vLLM 0.19.1 capability; "
        "the custom KV manager/runner patch is still required"
    )
    return CompatibilityReport(
        installed,
        installed == PINNED_VLLM_VERSION,
        scheduler_hooks,
        kv_hooks,
        native_layer_fork,
        tuple(details),
    )


def require_pinned_baseline() -> CompatibilityReport:
    report = inspect_vllm()
    if not report.baseline_ready:
        raise RuntimeError(
            "Unsupported vLLM installation: "
            f"expected={PINNED_VLLM_VERSION}, installed={report.installed_version}, "
            f"details={'; '.join(report.details)}"
        )
    return report
