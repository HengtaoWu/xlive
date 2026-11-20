import type { ActionToFunctionMap } from "./types";
const metaMap: ActionToFunctionMap = {
    "tool_called": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.tool_call = {
            name: data.name,
            args: data.args || {},
            calledAt: typeof data.called_at === "number"
                ? data.called_at
                : Date.now() / 1000,
        };
    }
};

export default metaMap;
