"""Small WebSocket bridge around the local Smart Turn v3 ONNX checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
import websockets


SAMPLE_RATE = 16_000
MAX_SAMPLES = 8 * SAMPLE_RATE


class SmartTurnModel:
    def __init__(self, model_path: Path, threshold: float) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Smart Turn model not found: {model_path}")
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        if not providers:
            raise RuntimeError("onnxruntime has no available execution provider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self.threshold = threshold
        print(
            "[smart-turn] ready "
            f"model={model_path} providers={self.session.get_providers()} "
            f"threshold={threshold}",
            flush=True,
        )

    def classify(self, pcm16: bytes) -> float:
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size > MAX_SAMPLES:
            audio = audio[-MAX_SAMPLES:]
        elif audio.size < MAX_SAMPLES:
            # Align the speech endpoint with the end of the eight-second window.
            audio = np.pad(audio, (MAX_SAMPLES - audio.size, 0))
        features = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        ).input_features
        output = self.session.run(None, {"input_features": features})[0]
        return float(np.asarray(output).reshape(-1)[0])


def _error(session_id: object, message: str) -> str:
    return json.dumps({"type": "error", "session_id": session_id, "error": message})


async def serve_connection(websocket: Any, model: SmartTurnModel) -> None:
    async for raw in websocket:
        session_id: object = None
        try:
            if not isinstance(raw, str):
                raise ValueError("expected JSON text")
            message = json.loads(raw)
            if not isinstance(message, dict) or message.get("type") != "classify":
                raise ValueError("expected classify request")
            session_id = message.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id is required")
            if message.get("sample_rate") != SAMPLE_RATE:
                raise ValueError("Smart Turn requires 16000 Hz audio")
            encoded = message.get("audio")
            if not isinstance(encoded, str):
                raise ValueError("audio is required")
            pcm16 = base64.b64decode(encoded, validate=True)
            if not pcm16 or len(pcm16) % 2:
                raise ValueError("audio must contain non-empty PCM16 samples")
            probability = await asyncio.to_thread(model.classify, pcm16)
            complete = probability >= model.threshold
            await websocket.send(
                json.dumps(
                    {
                        "type": "turn_state",
                        "session_id": session_id,
                        "state": {
                            "state": "speak" if complete else "nonidle",
                            "confidence": probability,
                        },
                        "ts": time.time(),
                    }
                )
            )
        except Exception as exc:
            await websocket.send(_error(session_id, str(exc)))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    model = SmartTurnModel(args.model, args.threshold)
    async with websockets.serve(
        lambda ws: serve_connection(ws, model),
        args.host,
        args.port,
        max_size=2**22,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
