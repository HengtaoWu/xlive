import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import type { SessionServiceURLConfig } from "../bases/http";
import { Conversation } from "../conversation";

export type {
    AudioChunkCallback,
    Session,
    SessionConfig,
    SessionDetail,
    SessionState,
    SessionSummary,
    TaskMessage,
};

/** Current browser-side conversation snapshot. */
type SessionState = Conversation["state"];

type AudioChunkCallback = (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void;

/** One entry returned by the persisted-session history endpoint. */
type SessionSummary = {
    session_id: string;
    title: string | null;
    updated_at?: string | null;
};

type TaskMessage = {
    id: string;
    role: "user" | "assistant" | "tool";
    content: string;
    created_at: number;
};

/** Restored messages and tasks for a persisted session. */
type SessionDetail = {
    session_id: string;
    title: string | null;
    messages: Array<{
        role: "user" | "assistant" | "info";
        content: string;
    }>;
    tasks?: Array<{
        task_id: string;
        title?: string;
        backend?: string;
        status: "running" | "completed" | "failed" | "cancelled";
        detail?: string;
        updates?: string[];
        created_at?: number;
        updated_at?: number;
        completed_at?: number | null;
        whiteboard?: Conversation["state"]["whiteboard"] | null;
    }>;
};

interface SessionConfig {
    inputConfig?: Partial<InputAudioSessionConfig>;
    outputConfig?: Partial<OutputAudioSessionConfig>;
    serviceURLs?: SessionServiceURLConfig;
}

/** Controller for one live or restored X-Live session. */
interface Session {
    open(): Promise<void>;
    close(): Promise<void>;
    onStateChange(callback: (state: Conversation["state"]) => void): void;
    readonly state: Conversation["state"];
    onInputAudioChunk(callback: AudioChunkCallback): void;
    onOutputAudioChunk(callback: AudioChunkCallback): void;
    onFullAudioChunk(callback: AudioChunkCallback): void;
    muted: boolean;
    changeVoice(voiceName: string): Promise<void>;
    /** Cancels the task attached to UI such as an open whiteboard. */
    cancelTask(taskId: string): Promise<void>;
    uploadFile(file: Blob, endpoint?: string | URL): Promise<void>;
    getSessions(): Promise<SessionSummary[]>;
    getTaskMessages(taskId: string): Promise<TaskMessage[]>;
    /** Pass `null` to leave history and start a fresh session. */
    switchSession(sessionId: string | null): Promise<void>;
    renameSession(sessionId: string, title: string): Promise<SessionSummary>;
    deleteSession(sessionId: string): Promise<void>;
}
