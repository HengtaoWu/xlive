import { BaseHTTPClient } from "../bases/http";
import type { ResolvableURL, SessionServiceURLs } from "../bases/http";
import { Conversation } from "../conversation";
import type { ConversationMessage, ConversationTask } from "../conversation";
import type { SessionDetail, SessionSummary, TaskMessage } from "./types";

export { createSessionAPI };

type AuthorizedOperation = <T>(operation: (token: string) => Promise<T>) => Promise<T>;

function mapSessionMessages(messages: SessionDetail["messages"]): ConversationMessage[] {
    return messages.map((message) => ({
        role: message.role,
        content: message.content,
        final: true,
    }));
}

function mapSessionTasks(tasks: SessionDetail["tasks"] = []): ConversationTask[] {
    return tasks.map((task) => {
        const startedAt = task.created_at
            ? new Date(task.created_at * 1000).toISOString()
            : new Date().toISOString();
        const updatedAt = task.updated_at
            ? new Date(task.updated_at * 1000).toISOString()
            : startedAt;
        const mapped: ConversationTask = {
            id: task.task_id,
            title: task.title || "Background task",
            backend: task.backend || "Codex",
            status: task.status,
            detail: task.detail || "",
            updates: task.updates ?? [],
            startedAt,
            updatedAt,
        };
        if (task.completed_at) {
            mapped.completedAt = new Date(task.completed_at * 1000).toISOString();
        }
        return mapped;
    });
}

function createSessionAPI(
    {
        closeRuntime,
        conversation,
        ensureLoggedIn,
        httpClient,
        serviceURLs,
        withAuthorizedToken,
    }: {
        closeRuntime: () => Promise<void>;
        conversation: Conversation;
        ensureLoggedIn: () => Promise<string>;
        httpClient: BaseHTTPClient;
        serviceURLs: SessionServiceURLs;
        withAuthorizedToken: AuthorizedOperation;
    },
) {
    function authorizedGetJSON<T>(input: ResolvableURL): Promise<T> {
        return withAuthorizedToken((token) => httpClient.getJSON<T>(input, token));
    }

    function loadSessionDetail(sessionId: string): Promise<SessionDetail> {
        return authorizedGetJSON<SessionDetail>(serviceURLs.sessionDetail(sessionId));
    }

    return {
        async getTaskMessages(taskId: string): Promise<TaskMessage[]> {
            const sessionId = conversation.state.sessionId;
            if (!sessionId) {
                throw new Error("No session selected");
            }
            const base = serviceURLs.sessionDetail(sessionId);
            const address = String(base).replace(/\/$/, "") + `/tasks/${encodeURIComponent(taskId)}/messages`;
            const payload = await authorizedGetJSON<{ messages: TaskMessage[] }>(address);
            return payload.messages;
        },
        async getSessions(): Promise<SessionSummary[]> {
            const payload = await authorizedGetJSON<{ sessions?: SessionSummary[] }>(
                serviceURLs.sessions,
            );
            return payload.sessions ?? [];
        },
        async switchSession(sessionId: string | null): Promise<void> {
            await ensureLoggedIn();
            await closeRuntime();
            if (!sessionId) {
                conversation.switch(null, []);
                return;
            }

            const payload = await loadSessionDetail(sessionId);
            conversation.switch(
                payload.session_id,
                mapSessionMessages(payload.messages),
                mapSessionTasks(payload.tasks),
            );
        },
        async renameSession(sessionId: string, title: string): Promise<SessionSummary> {
            const normalized = title.trim();
            if (!normalized) {
                throw new Error("Session title cannot be empty");
            }
            return await withAuthorizedToken((token) =>
                httpClient.patchJSON<SessionSummary>(
                    serviceURLs.sessionDetail(sessionId),
                    token,
                    { title: normalized },
                )
            );
        },
        async deleteSession(sessionId: string): Promise<void> {
            await withAuthorizedToken((token) =>
                httpClient.deleteJSON<{ session_id: string }>(
                    serviceURLs.sessionDetail(sessionId),
                    token,
                )
            );
        },
        async probeSessionRecovery(sessionId: string | null): Promise<boolean> {
            if (!sessionId) {
                return false;
            }
            try {
                await loadSessionDetail(sessionId);
                return true;
            } catch {
                return false;
            }
        },
        async refreshSession(sessionId: string): Promise<void> {
            const payload = await loadSessionDetail(sessionId);
            conversation.switch(
                payload.session_id,
                mapSessionMessages(payload.messages),
                mapSessionTasks(payload.tasks),
            );
        },
        async uploadFile(file: Blob, endpoint?: string | URL): Promise<void> {
            const sessionId = conversation.state.sessionId;
            if (!sessionId) {
                throw new Error("No session selected");
            }

            conversation.state.streamState = "processing";
            try {
                await withAuthorizedToken((token) =>
                    httpClient.postFile(endpoint ?? serviceURLs.upload, token, sessionId, file),
                );
            } finally {
                conversation.state.streamState = "idle";
            }
        },
    };
}
