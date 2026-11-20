import type { ActionToFunctionMap } from "./types";
import clientMap from "./client";
import messagesMap from "./messages";
import outputMap from "./output";
import sessionMap from "./session";
import clientOperationMap from "./client-operations";
import metaMap from "./meta";
import latencyMap from "./latency";
import inputMap from "./input";
import taskMap from "./tasks";
import whiteboardMap from "./whiteboard";

export { ACTION_TO_FUNCTION };

const ACTION_TO_FUNCTION: ActionToFunctionMap = {};

for (const handlers of [
    clientMap,
    messagesMap,
    outputMap,
    sessionMap,
    clientOperationMap,
    metaMap,
    latencyMap,
    inputMap,
    taskMap,
    whiteboardMap,
]) {
    for (const [action, handler] of Object.entries(handlers)) {
        if (handler) {
            ACTION_TO_FUNCTION[action] = handler;
        }
    }
}
