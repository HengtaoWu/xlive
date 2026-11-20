import type { ConversationMessage, ConversationTask } from "../conversation";
import type { ActionToFunctionMap } from "./types";

const sessionMap: ActionToFunctionMap = {
    "session_attached": async (data, websocket, conversation, outputAudioSession) => {
        const sid = typeof data?.session_id === "string" ? data.session_id : null;
        if (Array.isArray(data?.messages)) {
            const messages: ConversationMessage[] = [];
            for (const item of data.messages) {
                if (!item || typeof item !== "object") {
                    continue;
                }
                const role = item.role;
                const content = item.content;
                if (
                    (role === "user" || role === "assistant")
                    && typeof content === "string"
                    && content.trim()
                ) {
                    messages.push({ role, content, final: true });
                }
            }
            const tasks: ConversationTask[] = [];
            if (Array.isArray(data?.tasks)) {
                for (const item of data.tasks) {
                    if (!item || typeof item !== "object" || typeof item.task_id !== "string") {
                        continue;
                    }
                    const status = item.status;
                    if (!["running", "completed", "failed", "cancelled"].includes(status)) {
                        continue;
                    }
                    const startedAt = item.created_at
                        ? new Date(item.created_at * 1000).toISOString()
                        : new Date().toISOString();
                    const task: ConversationTask = {
                        id: item.task_id,
                        title: item.title || "Background task",
                        backend: item.backend || "Codex",
                        status,
                        detail: item.detail || "",
                        updates: Array.isArray(item.updates)
                            ? item.updates.filter((update: unknown): update is string => typeof update === "string")
                            : [],
                        startedAt,
                        updatedAt: item.updated_at
                            ? new Date(item.updated_at * 1000).toISOString()
                            : startedAt,
                    };
                    if (item.whiteboard && typeof item.whiteboard === "object") {
                        task.whiteboard = {
                            open: false,
                            taskId: item.task_id,
                            title: item.whiteboard.title || "Whiteboard",
                            version: Number(item.whiteboard.version) || 0,
                            blocks: Array.isArray(item.whiteboard.blocks)
                                ? item.whiteboard.blocks
                                : [],
                        };
                    }
                    if (item.completed_at) {
                        task.completedAt = new Date(item.completed_at * 1000).toISOString();
                    }
                    tasks.push(task);
                }
            }
            conversation.switch(sid, messages, tasks);
            return;
        }
        if (conversation.state.sessionId !== sid) {
            conversation.switch(sid, []);
            return;
        }
        conversation.state.sessionId = sid;
    },
};

export default sessionMap;
