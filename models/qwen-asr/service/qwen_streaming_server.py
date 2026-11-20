#!/usr/bin/env python3
"""Qwen3-ASR streaming WebSocket service for X-Live.

The wire protocol deliberately remains small and stable:
``start`` JSON -> PCM16 binary chunks -> ``partial`` JSON -> ``end`` JSON ->
``final`` JSON.  Qwen3-ASR owns transcript generation only; turn acceptance
continues to be the responsibility of the configured turn detector.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
from time import perf_counter
from typing import Any

import numpy as np
import websockets


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    return parser


@dataclass(slots=True)
class SessionState:
    state: Any
    started_at: float
    first_partial_latency: float | None = None


class QwenStreamingRecognizer:
    """One shared Qwen vLLM backend with independent per-WebSocket states."""

    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "qwen-asr[vllm] is required; create models/environments/qwen-asr.yml"
            ) from exc

        self._args = args
        logging.info("loading Qwen3-ASR streaming model from %s", args.model)
        self._model = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
        )
        # vLLM owns one engine. Serialising state updates prevents concurrent
        # WebSocket sessions from interleaving calls into its streaming API.
        self._inference_lock = asyncio.Lock()
        logging.info("Qwen3-ASR streaming model loaded")

    def new_session(self) -> SessionState:
        return SessionState(
            state=self._model.init_streaming_state(
                unfixed_chunk_num=self._args.unfixed_chunk_num,
                unfixed_token_num=self._args.unfixed_token_num,
                chunk_size_sec=self._args.chunk_size_sec,
            ),
            started_at=perf_counter(),
        )

    async def transcribe(self, session: SessionState, pcm16: bytes) -> str:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        async with self._inference_lock:
            await asyncio.to_thread(self._model.streaming_transcribe, audio, session.state)
        text = _state_text(session.state)
        if text and session.first_partial_latency is None:
            session.first_partial_latency = perf_counter() - session.started_at
        return text

    async def finish(self, session: SessionState) -> str:
        async with self._inference_lock:
            await asyncio.to_thread(self._model.finish_streaming_transcribe, session.state)
        text = _state_text(session.state)
        if text and session.first_partial_latency is None:
            session.first_partial_latency = perf_counter() - session.started_at
        return text


def _state_text(state: Any) -> str:
    text = getattr(state, "text", "")
    if not isinstance(text, str):
        raise RuntimeError("Qwen3-ASR streaming state returned non-string text")
    return text


async def _send(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


async def handle_connection(
    websocket: Any,
    recognizer: QwenStreamingRecognizer,
    sample_rate: int,
) -> None:
    session: SessionState | None = None
    logging.info("Qwen3-ASR client connected")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if session is None:
                    await _send(websocket, {"type": "error", "text": "session not started"})
                    continue
                if len(message) % 2:
                    await _send(websocket, {"type": "error", "text": "PCM16 must contain complete samples"})
                    continue
                try:
                    text = await recognizer.transcribe(session, message)
                    await _send(websocket, {"type": "partial", "text": text})
                except Exception as exc:
                    logging.exception("Qwen3-ASR partial inference failed")
                    await _send(websocket, {"type": "error", "text": str(exc)})
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "text": "control message must be JSON"})
                continue
            if not isinstance(payload, dict):
                await _send(websocket, {"type": "error", "text": "control message must be an object"})
                continue

            message_type = payload.get("type")
            if message_type == "start":
                client_rate = payload.get("sample_rate", sample_rate)
                if client_rate != sample_rate:
                    await _send(websocket, {"type": "error", "text": f"expected {sample_rate} Hz PCM16"})
                    continue
                session = recognizer.new_session()
                await _send(websocket, {"type": "started", "sample_rate": sample_rate})
            elif message_type == "end":
                if session is None:
                    await _send(websocket, {"type": "error", "text": "session not started"})
                    continue
                try:
                    text = await recognizer.finish(session)
                    await _send(
                        websocket,
                        {
                            "type": "final",
                            "text": text,
                            "first_partial_latency": session.first_partial_latency,
                        },
                    )
                except Exception as exc:
                    logging.exception("Qwen3-ASR final inference failed")
                    await _send(websocket, {"type": "error", "text": str(exc)})
                finally:
                    session = None
            elif message_type == "reset":
                session = recognizer.new_session()
                await _send(websocket, {"type": "reset_ok"})
            elif message_type == "ping":
                await _send(websocket, {"type": "pong"})
            else:
                await _send(websocket, {"type": "error", "text": f"unknown message type: {message_type}"})
    except websockets.ConnectionClosed:
        logging.info("Qwen3-ASR client disconnected")


async def main() -> None:
    args = get_parser().parse_args()
    if args.sample_rate != 16_000:
        raise ValueError("Qwen3-ASR X-Live service requires 16 kHz PCM16 input")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    recognizer = QwenStreamingRecognizer(args)
    async with websockets.serve(
        lambda websocket: handle_connection(websocket, recognizer, args.sample_rate),
        args.host,
        args.port,
        max_size=None,
    ):
        logging.info("Qwen3-ASR server listening on ws://%s:%s", args.host, args.port)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
