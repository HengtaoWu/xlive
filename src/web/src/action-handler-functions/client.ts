import type { ActionToFunctionMap } from "./types";
import { onVadSpeechStart, onVadSpeechEnd } from "./utils";
const clientMap: ActionToFunctionMap = {
    "client_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        // Smart Turn classifies the completed VAD segment server-side.
        // Keep playback intact until the runtime requests a server-owned stop_tts.
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_start" })
    },
    "client_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_end" })
    },
    "client_audio_chunk_started": async (data, websocket, conversation, outputAudioSession) => {
        // Report playback start even while VAD is open so the runtime can
        // treat a completed user turn as an interruption when appropriate.
        if (conversation.state.streamState !== "listening") {
            conversation.state.streamState = 'speaking';
        }
        websocket.sendJson({
            action: "tts_playback_started",
            ...(data && typeof data === "object" ? data : {}),
        });
    },
    "client_audio_playback_finished": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = 'idle';
        websocket.sendJson({
            action: "tts_playback_finished",
            ...(data && typeof data === "object" ? data : {}),
        })
    },
    "client_audio_chunk_played": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            action: "tts_chunk_played",
            ...(data && typeof data === "object" ? data : {}),
        })
    }
};

export default clientMap;
