import type { ActionHandlerFunction } from "./types";
export { onVadSpeechStart, onVadSpeechEnd };

const onVadSpeechStart: ActionHandlerFunction = (_data, _websocket, conversation) => {
    conversation.state.streamState = 'listening';
}

const onVadSpeechEnd: ActionHandlerFunction = (_data, _websocket, conversation) => {
    conversation.state.streamState = 'idle';
}
