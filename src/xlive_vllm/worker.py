"""Worker entry point selecting the X-Live vLLM v2 GPU runner."""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker

from .runner import XLiveGPUModelRunner


class XLiveWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise RuntimeError("X-Live requires VLLM_USE_V2_MODEL_RUNNER=1")
        self.model_runner = XLiveGPUModelRunner(self.vllm_config, self.device)
