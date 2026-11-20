export type TTSPlaybackIdentity = {
    generation_id: string;
    utterance_id: string;
};

export type TTSChunkMetadata = TTSPlaybackIdentity & {
    chunk_id: string;
    sample_rate: number;
};

export type DecodedTTSChunk = {
    pcm: ArrayBuffer;
    metadata: TTSChunkMetadata | null;
};

const TTS_CHUNK_MAGIC = [0x58, 0x4c, 0x54, 0x31]; // XLT1
const PREFIX_BYTES = 8;

export function decodeTTSChunkFrame(frame: ArrayBuffer): DecodedTTSChunk {
    const bytes = new Uint8Array(frame);
    const identified = (
        bytes.length >= TTS_CHUNK_MAGIC.length
        && TTS_CHUNK_MAGIC.every((value, index) => bytes[index] === value)
    );
    if (!identified) {
        return { pcm: frame, metadata: null };
    }
    if (bytes.length < PREFIX_BYTES) {
        throw new Error("Invalid identified TTS frame prefix");
    }

    const headerSize = new DataView(frame).getUint32(4, false);
    const pcmOffset = PREFIX_BYTES + headerSize;
    if (headerSize === 0 || pcmOffset >= bytes.length) {
        throw new Error("Invalid identified TTS frame length");
    }
    const rawHeader = new TextDecoder().decode(bytes.slice(PREFIX_BYTES, pcmOffset));
    const header: unknown = JSON.parse(rawHeader);
    if (!isTTSChunkMetadata(header)) {
        throw new Error("Invalid identified TTS chunk metadata");
    }
    return {
        pcm: frame.slice(pcmOffset),
        metadata: header,
    };
}

export function isTTSPlaybackIdentity(value: unknown): value is TTSPlaybackIdentity {
    if (typeof value !== "object" || value === null) {
        return false;
    }
    const candidate = value as Record<string, unknown>;
    return (
        typeof candidate.generation_id === "string"
        && candidate.generation_id.length > 0
        && typeof candidate.utterance_id === "string"
        && candidate.utterance_id.length > 0
    );
}

function isTTSChunkMetadata(value: unknown): value is TTSChunkMetadata {
    if (!isTTSPlaybackIdentity(value)) {
        return false;
    }
    const candidate = value as unknown as Record<string, unknown>;
    return (
        typeof candidate.chunk_id === "string"
        && candidate.chunk_id.length > 0
        && typeof candidate.sample_rate === "number"
        && Number.isInteger(candidate.sample_rate)
        && candidate.sample_rate > 0
    );
}
