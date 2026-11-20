import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Any


@dataclass
class BaseEvent:
    """Base dataclass for all X-Live events."""

    timestamp: float = field(init=False)
    session_id: str
    TYPE: ClassVar[str] = "base"

    def __post_init__(self):
        self.timestamp = time.time()

    @property
    def event_type(self) -> str:
        return self.TYPE


@dataclass
class WebSocketMessageReceived(BaseEvent):
    TYPE: ClassVar[str] = "websocket.message_received"
    message: str = ""


@dataclass
class AudioFrameReceived(BaseEvent):
    TYPE: ClassVar[str] = "audio.frame_received"
    audio_data: bytes
    sample_rate: int = 16000


@dataclass
class VADSpeechStart(BaseEvent):
    TYPE: ClassVar[str] = "vad.speech_start"
    origin: str = "client"


@dataclass
class VADSpeechEnd(BaseEvent):
    TYPE: ClassVar[str] = "vad.speech_end"
    origin: str = "client"


@dataclass
class ASRResultPartial(BaseEvent):
    TYPE: ClassVar[str] = "asr.result_partial"
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display
    speech_pause: bool = False


@dataclass
class ASRResultFinal(BaseEvent):
    # Emit when ready for generation
    TYPE: ClassVar[str] = "asr.result_final"
    text: str = ""
    display_text: str = ""  # Cleaned text for frontend display


@dataclass
class LLMFirstChunk(BaseEvent):
    """Event for first LLM chunk/tool call (measure first token latency)."""

    TYPE: ClassVar[str] = "llm.first_chunk"


@dataclass
class LLMFirstSentence(BaseEvent):
    """Event for first synthesizable sentence (measure sentence latency)."""

    TYPE: ClassVar[str] = "llm.sentence_ready"


@dataclass
class TTSStarted(BaseEvent):
    TYPE: ClassVar[str] = "tts.started"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TTSPlaybackStarted(BaseEvent):
    """Browser confirmed that an identified audio chunk began playback."""

    TYPE: ClassVar[str] = "tts.playback_started"
    generation_id: str = ""
    utterance_id: str = ""
    chunk_id: str = ""


@dataclass
class TTSStopped(BaseEvent):
    TYPE: ClassVar[str] = "tts.stopped"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TTSPaused(BaseEvent):
    TYPE: ClassVar[str] = "tts.paused"


@dataclass
class TTSResumed(BaseEvent):
    TYPE: ClassVar[str] = "tts.resumed"


@dataclass
class TTSFinished(BaseEvent):
    TYPE: ClassVar[str] = "tts.finished"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class ResponseUpdate(BaseEvent):
    """User-visible response text update.

    Attributes
    ----------
    text : str
        Current response text exposed to the user.
    """

    TYPE: ClassVar[str] = "response.update"
    text: str = ""
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class ResponseFinish(BaseEvent):
    """Final user-visible response text.

    Attributes
    ----------
    text : str
        Complete response text exposed to the user.
    """

    TYPE: ClassVar[str] = "response.finish"
    text: str = ""
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TaskCancelRequested(BaseEvent):
    """Request cancellation of an Agent task from a direct UI control."""

    TYPE: ClassVar[str] = "task.cancel_requested"
    task_id: str = ""


@dataclass
class WhiteboardUpdated(BaseEvent):
    """Versioned whiteboard state forwarded directly to the browser."""

    TYPE: ClassVar[str] = "whiteboard.updated"
    task_id: str = ""
    title: str = ""
    version: int = 0
    state: str = "open"
    operations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskStatusUpdated(BaseEvent):
    """User-visible lifecycle snapshot for one asynchronous backend task."""

    TYPE: ClassVar[str] = "task.status_updated"
    task_id: str = ""
    title: str = ""
    status: str = "running"
    detail: str = ""
    backend: str = "Codex"
    updates: list[str] = field(default_factory=list)


@dataclass
class TTSTextSynthesized(BaseEvent):
    """Text marker emitted after one synthesized text segment is fully produced.

    Attributes
    ----------
    text : str
        Text segment that was synthesized.
    audio_duration : float
        Estimated playback duration of the synthesized audio in milliseconds.
    """

    TYPE: ClassVar[str] = "tts.text_synthesized"
    text: str = ""
    audio_duration: float = 0.0


@dataclass
class TTSVoiceChange(BaseEvent):
    TYPE: ClassVar[str] = "tts.reference_audio_changed"
    voice_name: str = ""


@dataclass
class TTSSpeedChange(BaseEvent):
    TYPE: ClassVar[str] = "tts.speed_changed"
    speed: float = 1.0  # 0.5 ~ 1.5


@dataclass
class TTSChunkReady(BaseEvent):
    """Indicates one TTS audio chunk is ready for sending. Not emitted when the chunk is generated."""
    TYPE: ClassVar[str] = "tts.chunk_ready"
    audio_chunk: bytes = b""
    sample_rate: int = 48000
    generation_id: str = ""
    utterance_id: str = ""
    chunk_id: str = ""


@dataclass
class TTSChunkPlayed(BaseEvent):
    """Frontend confirmed playback completion for a TTS audio chunk.

    InputGateway publishes this after receiving tts_chunk_played so downstream
    listeners can observe frontend playback completion in FIFO order.
    """

    TYPE: ClassVar[str] = "tts.chunk_played_confirm"
    generation_id: str = ""
    utterance_id: str = ""
    chunk_id: str = ""


@dataclass
class TTSPlaybackFinished(BaseEvent):
    TYPE: ClassVar[str] = "tts.playback_finished"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class FullAudioFrameReady(BaseEvent):
    TYPE: ClassVar[str] = "audio.full_frame_ready"
    audio_chunk: bytes = b""
    sample_rate: int = 48000
    channels: int = 2
    format: str = "pcm_s16le"


@dataclass
class ErrorOccurred(BaseEvent):
    TYPE: ClassVar[str] = "error.occurred"
    error_type: str = ""
    error_message: str = ""


@dataclass
class ToolCallOccurred(BaseEvent):
    """LLM/Agent tool invocation notification."""

    TYPE: ClassVar[str] = "agent.tool_called"
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMAgentLoop(BaseEvent):
    """Request one agent-context accept loop iteration for the session."""

    TYPE: ClassVar[str] = "llm.agent_loop"


@dataclass
class LatencyMetricsUpdated(BaseEvent):
    """Fine-grained backend latency metrics (milliseconds)."""

    TYPE: ClassVar[str] = "metrics.latency_updated"
    network_latency_ms: int = 0  # VAD start → backend receives first frame
    asr_latency_ms: int = 0  # First frame → ASR final result
    llm_first_token_ms: int = 0  # ASR done → LLM first token
    llm_sentence_ms: int = 0  # ASR done → first synthesizable sentence
    tts_first_chunk_ms: int = 0  # LLM sentence done → first TTS chunk




@dataclass
class TurnTTSStartRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_start_requested"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TurnTTSPauseRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_pause_requested"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TurnTTSResumeRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_resume_requested"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TurnTTSStopRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_stop_requested"
    reason: str = ""  # e.g., verification_valid|playback_finished
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TurnTTSFlushRequested(BaseEvent):
    TYPE: ClassVar[str] = "turn.tts_flush_requested"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class TurnTTSTextAppendRequested(BaseEvent):
    """Request to append text into ongoing TTS stream (sim-trans)."""

    TYPE: ClassVar[str] = "turn.tts_text_append_requested"
    text: str = ""
    generation_id: str = ""
    utterance_id: str = ""




@dataclass
class ClockSyncReceived(BaseEvent):
    """Clock-sync event for offset calculation."""

    TYPE: ClassVar[str] = "clock.sync_received"
    client_send_ts: float = 0.0
    server_recv_ts: float = 0.0
    client_recv_ts: float = 0.0


@dataclass
class SessionReady(BaseEvent):
    """Browser audio sessions are ready for optional proactive output."""

    TYPE: ClassVar[str] = "session.ready"
    request_greeting: bool = False
    language: str = "zh"


@dataclass
class SessionConfigReceived(BaseEvent):
    """Client sent per-session runtime configuration."""

    TYPE: ClassVar[str] = "session.config_received"
