import type { ActionToFunctionMap } from "./types";

const taskMap: ActionToFunctionMap = {
    task_updated: async (data, _websocket, conversation) => {
        if (!data || typeof data.task_id !== "string") {
            return;
        }
        conversation.upsertTask(data);
    },
};

export default taskMap;
