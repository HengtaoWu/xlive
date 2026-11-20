import type { WhiteboardOperation } from "../conversation";
import type { ActionToFunctionMap } from "./types";

const validKinds = new Set(["heading", "text", "equation"]);

function parseOperations(value: unknown): WhiteboardOperation[] {
    if (!Array.isArray(value)) {
        return [];
    }
    const operations: WhiteboardOperation[] = [];
    for (const item of value) {
        if (!item || typeof item !== "object") {
            continue;
        }
        const operation = item as Record<string, unknown>;
        if (operation.type === "clear") {
            operations.push({ type: "clear" });
        } else if (
            operation.type === "append"
            && typeof operation.id === "string"
            && typeof operation.kind === "string"
            && validKinds.has(operation.kind)
            && typeof operation.content === "string"
        ) {
            operations.push({
                type: "append",
                id: operation.id,
                kind: operation.kind as "heading" | "text" | "equation",
                content: operation.content,
            });
        } else if (
            operation.type === "replace"
            && typeof operation.id === "string"
            && typeof operation.content === "string"
        ) {
            operations.push({ type: "replace", id: operation.id, content: operation.content });
        } else if (operation.type === "highlight" && typeof operation.id === "string") {
            operations.push({
                type: "highlight",
                id: operation.id,
                ...(typeof operation.highlighted === "boolean"
                    ? { highlighted: operation.highlighted }
                    : {}),
            });
        } else if (operation.type === "remove" && typeof operation.id === "string") {
            operations.push({ type: "remove", id: operation.id });
        }
    }
    return operations;
}

const whiteboardMap: ActionToFunctionMap = {
    whiteboard_updated: async (data, _websocket, conversation) => {
        if (
            !data
            || typeof data.task_id !== "string"
            || typeof data.version !== "number"
            || !Number.isSafeInteger(data.version)
            || data.version < 1
        ) {
            return;
        }
        const state = data.state === "open" || data.state === "closed"
            ? data.state
            : undefined;
        conversation.applyWhiteboardUpdate({
            task_id: data.task_id,
            version: data.version,
            ...(typeof data.title === "string" ? { title: data.title } : {}),
            ...(state ? { state } : {}),
            operations: parseOperations(data.operations),
        });
    },
};

export default whiteboardMap;
