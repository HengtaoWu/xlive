export { Conversation };
export type {
    ConversationMessage,
    ConversationState,
    ConversationTask,
    ConversationUser,
    ConversationWhiteboard,
    WhiteboardBlock,
    WhiteboardOperation,
};

type ConversationMessage = {
    role: "user" | "assistant" | "info";
    content: string;
    final?: boolean;
    generationId?: string;
}

type ConversationUser = {
    id: string;
}

type ConversationTask = {
    id: string;
    title: string;
    backend: string;
    status: "running" | "completed" | "failed" | "cancelled";
    detail: string;
    updates: string[];
    startedAt: string;
    updatedAt: string;
    completedAt?: string;
    whiteboard?: ConversationWhiteboard;
}

type WhiteboardBlock = {
    id: string;
    kind: "heading" | "text" | "equation";
    content: string;
    highlighted: boolean;
}

type WhiteboardOperation =
    | { type: "clear" }
    | { type: "append"; id: string; kind: WhiteboardBlock["kind"]; content: string }
    | { type: "replace"; id: string; content: string }
    | { type: "highlight"; id: string; highlighted?: boolean }
    | { type: "remove"; id: string };

type ConversationWhiteboard = {
    open: boolean;
    taskId: string | null;
    title: string;
    version: number;
    blocks: WhiteboardBlock[];
}

function cleanTaskUpdates(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value.filter(
        (item): item is string => typeof item === "string" && item.trim().length > 0
    );
}

function emptyWhiteboard(): ConversationWhiteboard {
    return {
        open: false,
        taskId: null,
        title: "Whiteboard",
        version: 0,
        blocks: [],
    };
}

function defaultConversation(): {
    connectionState: "connected" | "reconnecting" | "disconnected";
    streamState: "idle" | "listening" | "processing" | "speaking";
    sessionId: string | null;
    user: ConversationUser | null;
    latency: {
        network?: number,
        asr?: number,
        llmFirstToken?: number,
        llmSentence?: number,
        ttsFirstChunk?: number
    };
    messages: ConversationMessage[];
    tasks: ConversationTask[];
    whiteboard: ConversationWhiteboard;
    tool_call: {
        name: string,
        args: Record<string, any>,
        calledAt?: number,
    }
} {
    return {
        connectionState: "disconnected",
        streamState: "idle",
        sessionId: null,
        user: null,
        latency: {},
        messages: [],
        tasks: [],
        whiteboard: emptyWhiteboard(),
        tool_call: {
            name: "",
            args: {}
        }
    };
}

type ConversationState = ReturnType<typeof defaultConversation>;

class Conversation {
    private _state: ConversationState = defaultConversation();
    private readonly stateProxy: ConversationState = new Proxy(this._state, {
        set: (target, key: keyof ConversationState, value) => {
            Reflect.set(target, key, value);
            this.notifyStateChange();
            return true;
        },
    });
    private messagePrefixes: Array<string | undefined> = [];
    private generationMessageIndexes = new Map<string, number>();
    private generationPrefixes = new Map<string, string>();
    private stateChangeCallbacks = new Set<(state: ConversationState) => void>();
    private fullAudioChunkCallback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void = (_chunk, _sr) => { };

    private notifyStateChange(): void {
        for (const callback of this.stateChangeCallbacks) {
            callback(this._state);
        }
    }

    onStateChange(callback: (state: ConversationState) => void): void {
        callback(this._state);
        this.stateChangeCallbacks.add(callback);
    }

    onFullAudioChunk(
        callback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void
    ): void {
        this.fullAudioChunkCallback = callback;
    }

    get state(): ConversationState {
        return this.stateProxy;
    }

    setUser(user: ConversationUser | null): void {
        this._state.user = user;
        this.notifyStateChange();
    }

    switch(
        sessionId: string | null,
        messages: ConversationMessage[],
        tasks: ConversationTask[] = [],
    ): void {
        this._state.sessionId = sessionId;
        this._state.messages = [];
        for (const message of messages) {
            const restored = { ...message, final: true };
            const previous = this._state.messages[this._state.messages.length - 1];
            if (restored.role === "assistant" && previous?.role === "assistant") {
                previous.content = `${previous.content}\n${restored.content}`;
                continue;
            }
            this._state.messages.push(restored);
        }
        this._state.tasks = tasks.map((task) => ({ ...task }));
        this._state.whiteboard = emptyWhiteboard();
        this.messagePrefixes = this._state.messages.map(() => undefined);
        this.generationMessageIndexes.clear();
        this.generationPrefixes.clear();
        this._state.streamState = "idle";
        this._state.tool_call = {
            name: "",
            args: {}
        };
        this._state.latency = {};
        this.notifyStateChange();
    }

    appendMessage(message: ConversationMessage): void {
        if (message.role === "info") {
            this._state.messages.push(message);
            this.messagePrefixes.push(undefined);
            this.notifyStateChange();
            return;
        }

        if (message.role === "assistant" && message.generationId) {
            const existingIndex = this.generationMessageIndexes.get(
                message.generationId
            );
            if (existingIndex !== undefined) {
                const existing = this._state.messages[existingIndex]!;
                existing.content = `${this.generationPrefixes.get(message.generationId) ?? ""}${message.content}`;
                existing.final = message.final ?? false;
                this.messagePrefixes[existingIndex] = undefined;
                this.notifyStateChange();
                return;
            }
            const lastIndex = this._state.messages.length - 1;
            const previous = this._state.messages[this._state.messages.length - 1];
            if (previous?.role === "assistant") {
                const prefix = `${previous.content}\n`;
                previous.content = `${prefix}${message.content}`;
                previous.final = message.final ?? false;
                previous.generationId = message.generationId;
                this.generationMessageIndexes.set(message.generationId, lastIndex);
                this.generationPrefixes.set(message.generationId, prefix);
                this.messagePrefixes[lastIndex] = undefined;
                this.notifyStateChange();
                return;
            }
            this._state.messages.push({ ...message, final: message.final ?? false });
            this.messagePrefixes.push(undefined);
            this.generationMessageIndexes.set(
                message.generationId,
                this._state.messages.length - 1,
            );
            this.generationPrefixes.set(message.generationId, "");
            this.notifyStateChange();
            return;
        }

        const final = message.final ?? false;
        const lastIndex = this._state.messages.length - 1;
        const lastMessage = this._state.messages[lastIndex];

        if (lastMessage?.role === message.role && !lastMessage.final) {
            const prefix = this.messagePrefixes[lastIndex];
            if (prefix && message.content.startsWith(prefix)) {
                lastMessage.content = message.content.slice(prefix.length);
            } else {
                lastMessage.content = message.content;
                this.messagePrefixes[lastIndex] = undefined;
            }
            lastMessage.final = final;
            this.notifyStateChange();
            return;
        }

        const previousSameRole = this.findPreviousSameRoleMessage(message.role);
        let content = message.content;
        let prefix: string | undefined;

        if (
            previousSameRole &&
            !previousSameRole.message.final &&
            message.content.startsWith(previousSameRole.fullContent)
        ) {
            prefix = previousSameRole.fullContent;
            content = message.content.slice(prefix.length);
            if (final) {
                previousSameRole.message.final = true;
            }
            if (!content) {
                this.notifyStateChange();
                return;
            }
        }

        this._state.messages.push({
            role: message.role,
            content,
            final,
        });
        this.messagePrefixes.push(prefix);
        this.notifyStateChange();
    }

    private findPreviousSameRoleMessage(
        role: Exclude<ConversationMessage["role"], "info">,
    ): { message: ConversationMessage; fullContent: string } | undefined {
        for (let i = this._state.messages.length - 1; i >= 0; i--) {
            const message = this._state.messages[i];
            if (message?.role === role) {
                return {
                    message,
                    fullContent: `${this.messagePrefixes[i] ?? ""}${message.content}`,
                };
            }
        }
        return undefined;
    }

    updateLatency(latency: Conversation["state"]["latency"]): void {
        this._state.latency = { ...latency };
        this.notifyStateChange();
    }

    upsertTask(update: {
        task_id: string;
        title?: string;
        backend?: string;
        status: ConversationTask["status"];
        detail?: string;
        updates?: string[];
        updated_at?: number;
    }): void {
        const now = update.updated_at !== undefined
            ? new Date(update.updated_at * 1000).toISOString()
            : new Date().toISOString();
        const existing = this._state.tasks.find((task) => task.id === update.task_id);
        if (existing) {
            if (
                update.title
                && (!existing.title || existing.title === "Background task")
            ) {
                existing.title = update.title;
            }
            existing.backend = update.backend || existing.backend;
            existing.status = update.status;
            existing.detail = update.detail ?? existing.detail;
            if (update.updates !== undefined) {
                existing.updates = cleanTaskUpdates(update.updates);
            }
            existing.updatedAt = now;
            if (update.status !== "running") {
                existing.completedAt = now;
            }
        } else {
            const task: ConversationTask = {
                id: update.task_id,
                title: update.title || "Background task",
                backend: update.backend || "Codex",
                status: update.status,
                detail: update.detail ?? "",
                updates: cleanTaskUpdates(update.updates),
                startedAt: now,
                updatedAt: now,
            };
            if (update.status !== "running") {
                task.completedAt = now;
            }
            this._state.tasks.push(task);
        }
        this.notifyStateChange();
    }

    applyWhiteboardUpdate(update: {
        task_id: string;
        version: number;
        title?: string;
        state?: "open" | "closed";
        operations?: WhiteboardOperation[];
    }): void {
        const current = this._state.whiteboard;
        const sameTask = current.taskId === update.task_id;
        if (sameTask && update.version <= current.version) {
            return;
        }
        // A different task may only take ownership with an explicit open.
        // This prevents delayed operations from an older Agent task from
        // replacing the board currently visible to the user.
        if (!sameTask && update.state !== "open") {
            return;
        }

        const next: ConversationWhiteboard = sameTask
            ? { ...current, blocks: current.blocks.map((block) => ({ ...block })) }
            : { ...emptyWhiteboard(), taskId: update.task_id };
        next.version = update.version;
        if (update.title?.trim()) {
            next.title = update.title.trim();
        }
        if (update.state === "open") {
            next.open = true;
        } else if (update.state === "closed") {
            next.open = false;
        }

        for (const operation of update.operations ?? []) {
            if (operation.type === "clear") {
                next.blocks = [];
                continue;
            }
            if (operation.type === "append") {
                const existing = next.blocks.findIndex((block) => block.id === operation.id);
                const block: WhiteboardBlock = {
                    id: operation.id,
                    kind: operation.kind,
                    content: operation.content,
                    highlighted: false,
                };
                if (existing >= 0) {
                    next.blocks[existing] = block;
                } else {
                    next.blocks.push(block);
                }
                continue;
            }
            const index = next.blocks.findIndex((block) => block.id === operation.id);
            if (index < 0) {
                continue;
            }
            if (operation.type === "remove") {
                next.blocks.splice(index, 1);
            } else if (operation.type === "replace") {
                next.blocks[index] = { ...next.blocks[index]!, content: operation.content };
            } else if (operation.type === "highlight") {
                next.blocks[index] = {
                    ...next.blocks[index]!,
                    highlighted: operation.highlighted ?? true,
                };
            }
        }
        this._state.whiteboard = next;
        const task = this._state.tasks.find((item) => item.id === update.task_id);
        if (task) {
            task.whiteboard = {
                ...next,
                blocks: next.blocks.map((block) => ({ ...block })),
            };
        }
        this.notifyStateChange();
    }

    emitFullAudioChunk(pcmChunkInt16: ArrayBuffer, sampleRate: number): void {
        this.fullAudioChunkCallback(pcmChunkInt16, sampleRate);
    }
}
