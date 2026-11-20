"""Thin streaming API over the PyTorch Breeze inference runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from .templates import get_template, prepare_inputs, select_template_name
from .model.fast_streaming import (
    FastBreezeStreamingRuntime,
    FastStreamingChunk,
    FastStreamingConfig,
)
from .model.warmup_profile import load_warmup_profile

PACKAGE_ROOT = Path(__file__).resolve().parent
FAST_CONFIG = PACKAGE_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
OPTIONAL_AUDIO_FILE = File(None)


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool
    default_reference_audio: Path | None
    default_reference_text: str


_settings: ApiSettings | None = None
_request_lock = asyncio.Lock()
_STREAM_END = object()


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


def _next_stream_chunk(iterator: Iterator[FastStreamingChunk]):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


def _iter_seeded_audio_chunks(
    runtime: FastBreezeStreamingRuntime,
    inputs: dict[str, object],
    *,
    request_id: str,
    seed: int,
) -> Iterator[FastStreamingChunk]:
    """Start model sampling from the request seed, after all input preparation.

    The response body runs after the endpoint has returned a ``StreamingResponse``.
    Seeding only while preparing the request leaves model sampling vulnerable to
    lazy initialization or unrelated RNG use between preparation and iteration.
    The API is deliberately single-request, so resetting the process generators at
    this boundary gives each request an isolated, reproducible sampling start.
    """
    set_all_seeds(seed)
    token_digest = None
    token_frames = 0
    started_at = time.perf_counter()
    first_audio_at: float | None = None
    previous_chunk_at: float | None = None
    max_chunk_gap_seconds = 0.0
    queued_audio_seconds = 0.0
    max_underflow_seconds = 0.0
    completed = False
    if os.environ.get("BREEZE_DEBUG_TOKEN_HASH") == "1":
        token_digest = hashlib.sha256()

    def observe_token_frame(frame) -> None:
        nonlocal token_frames
        assert token_digest is not None
        token_digest.update(
            frame.detach().to(device="cpu").contiguous().numpy().tobytes()
        )
        token_frames += 1

    try:
        for chunk in runtime.iter_audio_chunks(
            inputs,
            request_id=request_id,
            seed=seed,
            token_observer=observe_token_frame if token_digest is not None else None,
        ):
            emitted_at = time.perf_counter()
            if first_audio_at is None:
                first_audio_at = emitted_at
            else:
                assert previous_chunk_at is not None
                max_chunk_gap_seconds = max(
                    max_chunk_gap_seconds,
                    emitted_at - previous_chunk_at,
                )
                max_underflow_seconds = max(
                    max_underflow_seconds,
                    emitted_at - first_audio_at - queued_audio_seconds,
                )
            queued_audio_seconds += float(chunk.audio.size) / chunk.sample_rate
            previous_chunk_at = emitted_at
            yield chunk
        completed = True
    finally:
        finished_at = time.perf_counter()
        if first_audio_at is not None and queued_audio_seconds > 0:
            elapsed_seconds = finished_at - started_at
            print(
                "breeze stream metrics: "
                f"request_id={request_id} completed={completed} "
                f"ttfa_ms={(first_audio_at - started_at) * 1000.0:.2f} "
                f"audio_seconds={queued_audio_seconds:.3f} "
                f"elapsed_seconds={elapsed_seconds:.3f} "
                f"rtf={elapsed_seconds / queued_audio_seconds:.3f} "
                f"max_chunk_gap_ms={max_chunk_gap_seconds * 1000.0:.2f} "
                f"max_underflow_ms={max_underflow_seconds * 1000.0:.2f}",
                flush=True,
            )
        if token_digest is not None:
            print(
                "breeze token trace: "
                f"request_id={request_id} seed={seed} frames={token_frames} "
                f"sha256={token_digest.hexdigest()}",
                flush=True,
            )


def _warmup_plain_stream(
    tokenizer,
    model,
    audio_tokenizer,
    runtime: FastBreezeStreamingRuntime,
) -> None:
    """Exercise the plain-text request path before the first live request."""
    request_id = "startup-plain-warmup"
    seed = 42
    set_all_seeds(seed)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [{"id": request_id, "text": "Hello from X-Live.", "speaker": "S0"}],
        get_template("tts_plain"),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    started_at = time.perf_counter()
    iterator = runtime.iter_audio_chunks(inputs, request_id=request_id, seed=seed)
    try:
        first_chunk = next(iterator)
    finally:
        iterator.close()
    print(
        "plain stream warmup: "
        f"elapsed_ms={(time.perf_counter() - started_at) * 1000.0:.2f} "
        f"audio_samples={first_chunk.audio.size}",
        flush=True,
    )


def _warmup_reference_stream(
    tokenizer,
    model,
    audio_tokenizer,
    runtime: FastBreezeStreamingRuntime,
    reference_audio: Path,
    reference_text: str,
) -> None:
    """Exercise the bundled reference-voice path before live traffic."""
    request_id = "startup-reference-warmup"
    seed = 42
    set_all_seeds(seed)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [
            {
                "id": request_id,
                "text": "你好。",
                "speaker": "S0",
                "ref_audio_path": str(reference_audio),
                "ref_text": reference_text,
            }
        ],
        get_template("ref_clone_tata"),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    started_at = time.perf_counter()
    iterator = runtime.iter_audio_chunks(inputs, request_id=request_id, seed=seed)
    try:
        first_chunk = next(iterator)
    finally:
        iterator.close()
    print(
        "reference stream warmup: "
        f"elapsed_ms={(time.perf_counter() - started_at) * 1000.0:.2f} "
        f"audio_samples={first_chunk.audio.size}",
        flush=True,
    )


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="breeze_ref_", suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            payload = await upload.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            temporary.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model,
        device=resolve_device(),
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)
        _warmup_plain_stream(tokenizer, model, audio_tokenizer, runtime)
        if settings.default_reference_audio is not None:
            _warmup_reference_stream(
                tokenizer,
                model,
                audio_tokenizer,
                runtime,
                settings.default_reference_audio,
                settings.default_reference_text,
            )

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.audio_tokenizer = audio_tokenizer
    app.state.runtime = runtime


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    _load_app(app, _settings)
    yield


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not hasattr(app.state, "runtime"):
        return JSONResponse({"status": "loading"}, status_code=503)
    runtime = app.state.runtime
    return JSONResponse(
        {
            "status": "ok",
            "sample_rate": runtime.sample_rate,
            "streaming": True,
            "fast_path": runtime.fast_enabled,
            "codec_chunk_frames": runtime.codec_chunk_frames,
            "default_reference_voice": (
                _settings is not None and _settings.default_reference_audio is not None
            ),
        }
    )


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    instruction: str | None = Form(None),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
) -> StreamingResponse:
    reference_path: Path | None = None
    delete_reference_path = False
    try:
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)
            delete_reference_path = True
        elif _settings is not None and _settings.default_reference_audio is not None:
            reference_path = _settings.default_reference_audio
            ref_text = _settings.default_reference_text

        request_id = f"api-{uuid.uuid4().hex}"
        instruction = instruction.strip() if instruction else None
        request = {
            "id": request_id,
            "text": text,
            "speaker": "S0",
        }
        if instruction:
            request["instruction"] = instruction
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = ref_text
        template_name = select_template_name(request)

    except Exception:
        if delete_reference_path and reference_path is not None:
            reference_path.unlink(missing_ok=True)
        raise

    async def body() -> AsyncIterator[bytes]:
        iterator: Iterator[FastStreamingChunk] | None = None
        try:
            # Own serialization inside the actual response stream. A request
            # cancelled before its body starts therefore cannot leak the lock.
            async with _request_lock:
                set_all_seeds(seed)
                inputs = await asyncio.to_thread(
                    prepare_inputs,
                    app.state.tokenizer,
                    app.state.audio_tokenizer,
                    app.state.model,
                    [request],
                    get_template(template_name),
                    guidance_scale=cfg_scale,
                    guidance_scale_ref=None,
                    guidance_scale_ins=None,
                )
                iterator = _iter_seeded_audio_chunks(
                    app.state.runtime,
                    inputs,
                    request_id=request_id,
                    seed=seed,
                )
                while True:
                    next_task = asyncio.create_task(
                        asyncio.to_thread(_next_stream_chunk, iterator)
                    )
                    try:
                        chunk = await asyncio.shield(next_task)
                    except asyncio.CancelledError:
                        # Closing while next() is executing in another thread is
                        # unsafe. Wait for that boundary, then release cleanly.
                        await asyncio.gather(next_task, return_exceptions=True)
                        raise
                    if chunk is _STREAM_END:
                        break
                    pcm = _pcm16(chunk.audio)
                    if pcm:
                        yield pcm
        finally:
            if iterator is not None:
                await asyncio.to_thread(iterator.close)

    def cleanup_reference() -> None:
        if delete_reference_path and reference_path is not None:
            reference_path.unlink(missing_ok=True)

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(app.state.runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "X-Streaming": "true",
            "Cache-Control": "no-store",
        },
        background=BackgroundTask(cleanup_reference),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--default-reference-audio", type=Path)
    parser.add_argument("--default-reference-text", default="")
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if (args.default_reference_audio is None) != (
        not args.default_reference_text.strip()
    ):
        parser.error(
            "--default-reference-audio and --default-reference-text must be set together"
        )
    if (
        args.default_reference_audio is not None
        and not args.default_reference_audio.is_file()
    ):
        parser.error(
            f"default reference audio not found: {args.default_reference_audio}"
        )

    global _settings
    _settings = ApiSettings(
        model=args.model,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        default_reference_audio=args.default_reference_audio,
        default_reference_text=args.default_reference_text.strip(),
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
