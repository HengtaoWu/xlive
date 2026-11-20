import type { ActionToFunctionMap } from "./types";
import { isTTSPlaybackIdentity } from "../tts-wire";
const outputMap: ActionToFunctionMap = {
    "start_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.beginTTS(
            isTTSPlaybackIdentity(data) ? data : null,
        );
    },
    "pause_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "stop_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.stopTTS(
            isTTSPlaybackIdentity(data) ? data : null,
        );
    },
    "resume_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
    "tts_finished": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.notifyTTSFinished(
            isTTSPlaybackIdentity(data) ? data : null,
        );
    },
};

export default outputMap;
