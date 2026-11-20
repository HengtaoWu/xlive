"""Binary WebSocket envelope for one identified TTS PCM chunk."""

from __future__ import annotations

import json
import struct


TTS_CHUNK_MAGIC = b"XLT1"


def encode_tts_chunk_frame(
    audio_chunk: bytes,
    *,
    generation_id: str,
    utterance_id: str,
    chunk_id: str,
    sample_rate: int,
) -> bytes:
    """Pack metadata and PCM into one ordered WebSocket binary message."""

    if not audio_chunk:
        raise ValueError("audio_chunk must be non-empty")
    if not generation_id or not utterance_id or not chunk_id:
        raise ValueError("identified TTS chunks require all identity fields")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    header = json.dumps(
        {
            "generation_id": generation_id,
            "utterance_id": utterance_id,
            "chunk_id": chunk_id,
            "sample_rate": sample_rate,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return TTS_CHUNK_MAGIC + struct.pack(">I", len(header)) + header + audio_chunk
