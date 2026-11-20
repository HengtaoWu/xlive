import "@fontsource-variable/geist/wght.css";
import "katex/dist/katex.min.css";
import "./styles.css";
import xliveLogoURL from "../../../../docs/assets/images/xlive-logo.png";
import katex from "katex";

import type { Session, SessionState, SessionSummary } from "../../src/session/types";
import { PCMLevelMeter } from "./audio-level";
import { resolveAppConfig } from "./app-config";
import {
    AudioReactiveLiquidOrb,
    type OrbConnectionState,
    type OrbSpeaker,
} from "./liquid-orb";

type ActivityTask = {
    id: string;
    sessionId: string;
    title: string;
    backend: string;
    status: "running" | "completed" | "failed" | "cancelled";
    startedAt: string;
    completedAt?: string | null;
    updates: string[];
    whiteboard: SessionState["whiteboard"] | undefined;
};

const appRoot = document.querySelector<HTMLElement>("#app");
if (!appRoot) {
    throw new Error("X-Live app root is missing");
}

appRoot.innerHTML = `
    <div class="app-shell">
        <header class="topbar">
            <a class="brand" href="/" aria-label="X-Live home">
                <span class="brand-logo-frame" aria-hidden="true">
                    <img class="brand-logo" src="${xliveLogoURL}" alt="" />
                </span>
            </a>
            <div class="session-switcher" id="session-switcher" data-open="false">
                <button class="session-menu-button" id="session-menu-button" type="button" aria-expanded="false" aria-controls="session-popover">
                    <span class="session-menu-label">Sessions</span>
                    <span class="session-menu-chevron" aria-hidden="true">${iconChevron()}</span>
                </button>
                <div class="session-popover" id="session-popover" aria-hidden="true">
                    <div class="session-history-list" id="session-history-list"></div>
                    <p class="session-history-empty" id="session-history-empty">Connect X-Live to view session history.</p>
                </div>
            </div>
            <div class="topbar-actions">
                <div class="voice-toggle" id="voice-toggle" role="group" aria-label="Voice language" data-language="zh">
                    <span class="voice-toggle-thumb" id="voice-toggle-thumb" aria-hidden="true"></span>
                    <button type="button" data-voice="zh" aria-label="Chinese voice" aria-pressed="true">中文</button>
                    <button type="button" data-voice="en" aria-label="English voice" aria-pressed="false">EN</button>
                </div>
                <a class="header-link demo-link" href="https://hengtaowu.github.io/xlive/" target="_blank" rel="noreferrer" aria-label="Open X-Live demo" title="Demo">
                    <span class="header-link-icon" aria-hidden="true">${iconDemo()}</span>
                </a>
                <a class="header-link github-link" id="github-link" href="https://github.com/HengtaoWu/xlive" target="_blank" rel="noreferrer" aria-label="Open X-Live on GitHub" title="GitHub">
                    <span class="header-link-icon" aria-hidden="true">${iconGitHub()}</span>
                </a>
            </div>
        </header>

        <main class="voice-workspace">
            <section class="transcript-panel" aria-labelledby="transcript-title">
                <div class="panel-heading">
                    <div class="panel-heading-copy">
                        <h2 id="transcript-title">Conversation</h2>
                    </div>
                </div>
                <div class="transcript-scroll" id="transcript-scroll" aria-live="polite">
                    <div class="empty-conversation" id="empty-conversation">
                        <p>Your conversation will appear here when a live session is connected.</p>
                    </div>
                    <div class="message-list" id="message-list"></div>
                </div>
            </section>

            <section class="orb-column" aria-label="X-Live voice presence">
                <div class="orb-stage" id="orb-stage" data-speaker="idle" data-connection="disconnected">
                    <div class="orb-atmosphere" aria-hidden="true"></div>
                    <div class="orb-css-fallback" aria-hidden="true">
                        <span></span>
                    </div>
                </div>
                <div class="tool-activity" id="tool-activity" aria-live="polite" hidden></div>
                <span class="visually-hidden" id="live-status-label" role="status">Microphone is off</span>
            </section>

            <aside class="right-rail" aria-label="Session and Agent information">
                <section class="activity-panel" id="activity-panel" aria-labelledby="activity-title">
                    <div class="panel-heading">
                        <h2 id="activity-title">Tasks</h2>
                    </div>
                    <div class="activity-list" id="activity-list"></div>
                    <p class="activity-empty" id="activity-empty">No tasks in this session.</p>
                </section>
            </aside>
        </main>

        <section class="whiteboard-window" id="whiteboard-window" aria-labelledby="whiteboard-title" hidden>
            <header class="whiteboard-header" id="whiteboard-header">
                <span class="whiteboard-mark" aria-hidden="true"></span>
                <div class="whiteboard-heading">
                    <span>Live whiteboard</span>
                    <h2 id="whiteboard-title">Whiteboard</h2>
                </div>
                <button class="whiteboard-hide" id="whiteboard-hide" type="button" aria-label="Close whiteboard and stop lesson" title="Close whiteboard and stop lesson">
                    ${iconClose()}
                </button>
            </header>
            <div class="whiteboard-canvas" id="whiteboard-canvas">
                <div class="whiteboard-content" id="whiteboard-content"></div>
                <p class="whiteboard-empty" id="whiteboard-empty">The teacher is preparing the board…</p>
            </div>
            <footer class="whiteboard-footer">
                <span class="whiteboard-live-dot" aria-hidden="true"></span>
                <span>Synced with the active lesson</span>
            </footer>
        </section>

        <dialog class="task-dialog" id="task-dialog" aria-labelledby="task-dialog-title">
            <header class="task-dialog-header" id="task-dialog-header">
                <div><span class="task-dialog-agent" id="task-dialog-agent">Codex</span><h2 id="task-dialog-title"></h2></div>
                <button class="task-dialog-close" id="task-dialog-close" type="button" aria-label="Close task details">${iconClose()}</button>
            </header>
            <div class="task-messages" id="task-messages" role="log" aria-label="Task conversation"></div>
        </dialog>

        <footer class="session-controls" id="session-controls" aria-label="Session controls">
            <button class="control-button conversation-button" id="conversation-button" type="button" aria-label="Start conversation">
                <span class="control-icon start-conversation-icon" aria-hidden="true">
                    <i></i><i></i><i></i><i></i><i></i>
                </span>
                <span class="control-icon end-conversation-icon">${iconEnd()}</span>
            </button>
            <button class="control-button microphone-button" id="microphone-button" type="button" aria-label="Mute microphone" aria-pressed="true" hidden>
                <span class="control-icon mic-on-icon">${iconMicrophone()}</span>
                <span class="control-icon mic-off-icon">${iconMicrophoneOff()}</span>
            </button>
        </footer>

    </div>
`;

const elements = {
    activityEmpty: requireElement("activity-empty"),
    activityList: requireElement("activity-list"),
    conversationButton: requireButton("conversation-button"),
    emptyConversation: requireElement("empty-conversation"),
    githubLink: requireAnchor("github-link"),
    voiceToggle: requireElement("voice-toggle"),
    voiceThumb: requireElement("voice-toggle-thumb"),
    liveStatusLabel: requireElement("live-status-label"),
    messageList: requireElement("message-list"),
    microphoneButton: requireButton("microphone-button"),
    orbStage: requireElement("orb-stage"),
    sessionControls: requireElement("session-controls"),
    sessionHistoryEmpty: requireElement("session-history-empty"),
    sessionHistoryList: requireElement("session-history-list"),
    sessionMenuButton: requireButton("session-menu-button"),
    sessionPopover: requireElement("session-popover"),
    sessionSwitcher: requireElement("session-switcher"),
    transcriptScroll: requireElement("transcript-scroll"),
    toolActivity: requireElement("tool-activity"),
    taskDialog: requireElement("task-dialog") as HTMLDialogElement,
    taskDialogClose: requireButton("task-dialog-close"),
    taskDialogHeader: requireElement("task-dialog-header"),
    taskDialogTitle: requireElement("task-dialog-title"),
    taskDialogAgent: requireElement("task-dialog-agent"),
    taskMessages: requireElement("task-messages"),
    whiteboardContent: requireElement("whiteboard-content"),
    whiteboardEmpty: requireElement("whiteboard-empty"),
    whiteboardHeader: requireElement("whiteboard-header"),
    whiteboardHide: requireButton("whiteboard-hide"),
    whiteboardTitle: requireElement("whiteboard-title"),
    whiteboardWindow: requireElement("whiteboard-window"),
};

class XLiveApp {
    private readonly config = resolveAppConfig();
    private selectedTaskId: string | null = null;
    private taskDetailsRevision = 0;
    private taskDetailsTimer = 0;
    private taskDrag: { pointerId: number; offsetX: number; offsetY: number } | null = null;
    private taskMessagesSignature = "";
    private readonly orb = new AudioReactiveLiquidOrb(elements.orbStage);
    private readonly inputMeter = new PCMLevelMeter((level) => this.handleInputLevel(level));
    private readonly outputMeter = new PCMLevelMeter((level) => this.handleOutputLevel(level));
    private session: Session | null = null;
    private voiceLanguage = "zh";
    private voiceDrag: {
        pointerId: number;
        startX: number;
        origin: number;
        position: number;
        moved: boolean;
        language: string;
    } | null = null;
    private sessionPreparation: Promise<Session | null> | null = null;
    private inputLevel = 0;
    private outputLevel = 0;
    private microphoneOn = false;
    private conversationStarted = false;
    private tasks: ActivityTask[] = [];
    private sessionHistory: SessionSummary[] = [];
    private sessionHistoryState: "idle" | "loading" | "ready" | "error" = "idle";
    private destroyed = false;
    private lastStatus = "";
    private readonly messageExitTimers = new Map<HTMLElement, number>();
    private scrollAnimationFrame = 0;
    private taskClockTimer = 0;
    private lastToolSignature = "";
    private toolActivityTimer = 0;
    private dismissedWhiteboardTaskId: string | null = null;
    private whiteboardDrag: {
        pointerId: number;
        offsetX: number;
        offsetY: number;
    } | null = null;
    private readonly renderedWhiteboardBlocks = new Set<string>();

    constructor() {
        elements.githubLink.href = this.config.githubURL;
        elements.taskDialogClose.addEventListener("click", () => elements.taskDialog.close());
        elements.taskDialog.addEventListener("close", this.closeTaskDetails);
        elements.taskDialogHeader.addEventListener("pointerdown", this.startTaskDrag);
        elements.taskDialogHeader.addEventListener("pointermove", this.moveTaskDrag);
        elements.taskDialogHeader.addEventListener("pointerup", this.endTaskDrag);
        elements.taskDialogHeader.addEventListener("pointercancel", this.endTaskDrag);
        elements.taskDialogHeader.addEventListener("lostpointercapture", this.endTaskDrag);
        window.addEventListener("resize", this.keepTaskInView);
        elements.voiceToggle.addEventListener("click", this.changeVoice);
        elements.voiceToggle.addEventListener("pointerdown", this.startVoiceDrag);
        elements.voiceToggle.addEventListener("pointermove", this.moveVoiceDrag);
        elements.voiceToggle.addEventListener("pointerup", this.endVoiceDrag);
        elements.voiceToggle.addEventListener("pointercancel", this.cancelVoiceDrag);
        elements.voiceToggle.addEventListener("lostpointercapture", this.cancelVoiceDrag);
        elements.conversationButton.addEventListener("click", this.handleConversationControl);
        elements.microphoneButton.addEventListener("click", this.toggleMicrophone);
        elements.sessionMenuButton.addEventListener("click", this.toggleSessionMenu);
        elements.sessionHistoryList.addEventListener("click", this.handleSessionHistoryClick);
        elements.sessionHistoryList.addEventListener("submit", this.handleSessionRenameSubmit);
        elements.whiteboardHide.addEventListener("click", this.hideWhiteboard);
        elements.whiteboardHeader.addEventListener("pointerdown", this.startWhiteboardDrag);
        window.addEventListener("pointermove", this.moveWhiteboard);
        window.addEventListener("pointerup", this.endWhiteboardDrag);
        window.addEventListener("pointercancel", this.endWhiteboardDrag);
        document.addEventListener("pointerdown", this.handleDocumentPointerDown);
        document.addEventListener("keydown", this.handleDocumentKeyDown);
        window.addEventListener("beforeunload", this.dispose, { once: true });
        this.renderMicrophone();
        this.renderTasks();
        this.taskClockTimer = window.setInterval(this.updateTaskDurations, 1000);
        this.renderSessionHistory();
        this.renderMessages([], "live:pending");
        void this.prepareLiveSession();
    }

    private renderVoice(): void {
        elements.voiceToggle.dataset.language = this.voiceLanguage;
        for (const button of elements.voiceToggle.querySelectorAll<HTMLButtonElement>("button[data-voice]")) {
            button.setAttribute("aria-pressed", String(button.dataset.voice === this.voiceLanguage));
        }
    }

    private startVoiceDrag = (event: PointerEvent): void => {
        if (event.button !== 0 || !event.isPrimary || this.voiceDrag) {
            return;
        }
        const bounds = elements.voiceToggle.getBoundingClientRect();
        const origin = this.voiceLanguage === "en" ? 1 : 0;
        this.voiceDrag = {
            pointerId: event.pointerId,
            startX: event.clientX,
            origin,
            position: origin,
            moved: false,
            language: event.clientX < bounds.left + bounds.width / 2 ? "zh" : "en",
        };
        elements.voiceToggle.setPointerCapture(event.pointerId);
        event.preventDefault();
    };

    private moveVoiceDrag = (event: PointerEvent): void => {
        const drag = this.voiceDrag;
        if (!drag || event.pointerId !== drag.pointerId) {
            return;
        }
        const distance = event.clientX - drag.startX;
        if (!drag.moved && Math.abs(distance) < 3) {
            return;
        }
        drag.moved = true;
        const travel = Math.max(1, elements.voiceThumb.offsetWidth);
        drag.position = Math.max(0, Math.min(1, drag.origin + distance / travel));
        elements.voiceToggle.classList.add("is-dragging");
        elements.voiceThumb.style.transform = `translateX(${drag.position * 100}%)`;
    };

    private endVoiceDrag = (event: PointerEvent): void => {
        if (!this.voiceDrag || event.pointerId !== this.voiceDrag.pointerId) {
            return;
        }
        this.moveVoiceDrag(event);
        const drag = this.voiceDrag;
        const language = drag.moved
            ? (event.clientX > drag.startX ? "en" : "zh")
            : drag.language;
        this.cancelVoiceDrag(event);
        void this.setVoiceLanguage(language);
    };

    private cancelVoiceDrag = (event: PointerEvent): void => {
        if (!this.voiceDrag || event.pointerId !== this.voiceDrag.pointerId) {
            return;
        }
        this.voiceDrag = null;
        elements.voiceToggle.classList.remove("is-dragging");
        elements.voiceThumb.style.transform = "";
        if (elements.voiceToggle.hasPointerCapture(event.pointerId)) {
            elements.voiceToggle.releasePointerCapture(event.pointerId);
        }
    };

    private changeVoice = async (event: MouseEvent): Promise<void> => {
        if (event.detail !== 0) {
            return;
        }
        const button = event.target instanceof Element
            ? event.target.closest<HTMLButtonElement>("button[data-voice]")
            : null;
        if (button?.dataset.voice) {
            await this.setVoiceLanguage(button.dataset.voice);
        }
    };

    private async setVoiceLanguage(language: string): Promise<void> {
        if ((language !== "zh" && language !== "en") || language === this.voiceLanguage) {
            return;
        }
        const previous = this.voiceLanguage;
        this.voiceLanguage = language;
        this.renderVoice();
        try {
            await this.session?.changeVoice(language);
        } catch (error) {
            this.voiceLanguage = previous;
            this.renderVoice();
            this.updateStatus(readableConnectionError(error));
        }
    }

    private handleConversationControl = async (): Promise<void> => {
        if (this.conversationStarted) {
            await this.endSession();
            return;
        }
        await this.connectLiveSession();
    };

    private toggleMicrophone = async (): Promise<void> => {
        if (!this.session) {
            await this.connectLiveSession();
            return;
        }
        this.microphoneOn = !this.microphoneOn;
        this.session.muted = !this.microphoneOn;
        if (!this.microphoneOn) {
            this.inputMeter.stop();
            this.handleInputLevel(0);
        }
        this.renderMicrophone();
        this.updateStatusFromLevels();
    };

    private handleInputLevel(level: number): void {
        this.inputLevel = level;
        if (this.outputLevel < 0.02) {
            this.setOrbLevel(level, level > 0.006 ? "user" : "idle");
        }
        this.updateStatusFromLevels();
    }

    private handleOutputLevel(level: number): void {
        this.outputLevel = level;
        if (level >= 0.02) {
            this.setOrbLevel(level, "agent");
        } else {
            this.setOrbLevel(this.inputLevel, this.inputLevel > 0.006 ? "user" : "idle");
        }
        this.updateStatusFromLevels();
    }

    private setOrbLevel(level: number, speaker: OrbSpeaker): void {
        this.orb.setState({ audioLevel: level, speaker });
    }

    private updateStatusFromLevels(): void {
        if (this.outputLevel >= 0.02) {
            this.updateStatus("X-Live is speaking");
        } else if (this.inputLevel >= 0.035) {
            this.updateStatus("Listening");
        } else {
            this.updateStatus(this.microphoneOn ? "Ready to listen" : "Microphone is muted");
        }
    }

    private updateStatus(label: string): void {
        if (label === this.lastStatus) {
            return;
        }
        this.lastStatus = label;
        elements.liveStatusLabel.textContent = label;
    }

    private connectLiveSession = async (): Promise<void> => {
        elements.conversationButton.disabled = true;
        this.microphoneOn = false;
        this.setConnectionState("connecting");
        this.updateStatus("Connecting to X-Live");

        try {
            const session = await this.prepareLiveSession();
            if (!session || this.destroyed) {
                return;
            }
            // The main control always starts a fresh conversation. Restoring a
            // saved conversation is handled only by an explicit history click.
            await session.switchSession(null);
            await session.open();
            if (this.session !== session) {
                await session.close();
                return;
            }
            this.microphoneOn = !session.muted;
            this.conversationStarted = true;
            this.setConnectionState("connected");
            this.renderMicrophone();
            this.updateStatus("Ready to listen");
            void this.refreshSessionHistory(session);
        } catch (error) {
            await this.session?.close().catch(() => undefined);
            this.conversationStarted = false;
            this.setConnectionState("error");
            this.updateStatus(readableConnectionError(error));
        } finally {
            elements.conversationButton.disabled = false;
        }
    };

    private prepareLiveSession = async (): Promise<Session | null> => {
        if (this.session) {
            return this.session;
        }
        if (this.sessionPreparation) {
            return await this.sessionPreparation;
        }

        this.sessionPreparation = this.createLiveSessionClient();
        try {
            return await this.sessionPreparation;
        } finally {
            this.sessionPreparation = null;
        }
    };

    private createLiveSessionClient = async (): Promise<Session | null> => {
        const endpoint = this.config.websocketURL.trim();
        if (!isWebSocketURL(endpoint)) {
            this.updateStatus("Live endpoint is not configured");
            this.sessionHistoryState = "error";
            this.renderSessionHistory();
            return null;
        }

        this.sessionHistoryState = "loading";
        this.renderSessionHistory();
        try {
            // Creating the SDK client and listing history use HTTP only. The
            // realtime socket and microphone remain closed until Start.
            const { createSession } = await import("../../src/index");
            const session = createSession(endpoint, {
                inputConfig: {
                    enableVAD: true,
                    enableEnhancer: false,
                    vadRedemptionMs: 280,
                },
                outputConfig: {
                    prebufferMs: 180,
                },
            });
            if (this.destroyed) {
                await session.close().catch(() => undefined);
                return null;
            }
            this.session = session;
            await session.changeVoice(this.voiceLanguage);
            session.onInputAudioChunk((chunk) => this.inputMeter.pushPCM16(chunk));
            session.onOutputAudioChunk((chunk) => this.outputMeter.pushPCM16(chunk));
            session.onStateChange((state) => this.handleSessionState(state));

            // Discard a locally remembered active conversation while retaining
            // its user token, so the landing page stays a clean new session.
            await session.switchSession(null);
            this.tasks = [];
            this.renderMessages([], "live:new");
            this.renderTasks();
            await this.refreshSessionHistory(session);
            if (this.session === session && !this.destroyed) {
                this.updateStatus("Ready to start");
            }
            return session;
        } catch (error) {
            const session = this.session;
            this.session = null;
            await session?.close().catch(() => undefined);
            this.sessionHistory = [];
            this.sessionHistoryState = "error";
            this.renderSessionHistory();
            this.updateStatus(readableConnectionError(error));
            return null;
        }
    };

    private handleSessionState(state: SessionState): void {
        const sessionId = state.sessionId ?? "live-current";
        this.tasks = state.tasks.map((task) => ({
            id: task.id,
            sessionId,
            title: task.title,
            backend: task.backend,
            status: task.status,
            startedAt: task.startedAt,
            completedAt: task.completedAt ?? null,
            updates: [...task.updates],
            whiteboard: task.whiteboard ? {
                ...task.whiteboard,
                blocks: task.whiteboard.blocks.map((block) => ({ ...block })),
            } : undefined,
        }));
        this.renderMessages(state.messages, `live:${state.sessionId ?? "current"}`);
        this.renderTasks();
        this.renderToolActivity(state.tool_call, state.tasks);
        this.renderWhiteboard(state.whiteboard);
        if (state.connectionState === "reconnecting") {
            this.setConnectionState("reconnecting");
            this.updateStatus("Reconnecting");
        } else if (state.connectionState === "connected") {
            this.setConnectionState("connected");
        } else {
            this.setConnectionState("disconnected");
            this.updateStatus("Connection closed");
        }
        if (state.streamState === "processing" && this.outputLevel < 0.02) {
            this.updateStatus("X-Live is thinking");
        } else if (state.streamState === "speaking" && this.outputLevel < 0.02) {
            this.updateStatus("X-Live is preparing audio");
        }
    }

    private renderMessages(messages: SessionState["messages"], source: string): void {
        const sourceChanged = elements.messageList.dataset.source !== source;
        const distanceFromBottom = elements.transcriptScroll.scrollHeight
            - elements.transcriptScroll.scrollTop
            - elements.transcriptScroll.clientHeight;
        const shouldFollowLatest = sourceChanged || distanceFromBottom < 56;
        const recentMessages = messages
            .map((message, index) => ({ message, index }))
            .filter(({ message }) => message.content.trim().length > 0);
        const desiredKeys = new Set(recentMessages.map(({ message, index }) => `${source}:${index}:${message.role}`));
        const existing = new Map<string, HTMLElement>();
        let addedMessage = false;
        for (const child of elements.messageList.children) {
            if (child instanceof HTMLElement && child.dataset.messageKey) {
                existing.set(child.dataset.messageKey, child);
            }
        }

        recentMessages.forEach(({ message, index }) => {
            const key = `${source}:${index}:${message.role}`;
            let article = existing.get(key);
            const isNew = !article;
            if (!article) {
                addedMessage = true;
                article = document.createElement("article");
                article.dataset.messageKey = key;
                const author = document.createElement("span");
                author.className = "message-author";
                const content = document.createElement("p");
                article.append(author, content);
            }

            const pendingExit = this.messageExitTimers.get(article);
            if (pendingExit !== undefined) {
                window.clearTimeout(pendingExit);
                this.messageExitTimers.delete(article);
            }
            article.className = `message message-${message.role}${message.final === false ? " message-streaming" : ""}`;
            article.removeAttribute("aria-hidden");
            article.querySelector<HTMLElement>(".message-author")!.textContent =
                message.role === "assistant" ? "X-Live" : message.role === "user" ? "You" : "Info";
            article.querySelector<HTMLParagraphElement>("p")!.textContent = message.content;
            if (isNew) {
                article.classList.add("message-entering");
            }
            elements.messageList.append(article);
        });

        for (const [key, article] of existing) {
            if (desiredKeys.has(key) || article.classList.contains("message-exiting")) {
                continue;
            }
            article.classList.add("message-exiting");
            article.setAttribute("aria-hidden", "true");
            const timer = window.setTimeout(() => {
                article.remove();
                this.messageExitTimers.delete(article);
            }, 420);
            this.messageExitTimers.set(article, timer);
        }
        elements.emptyConversation.hidden = recentMessages.length > 0;
        elements.messageList.dataset.source = source;
        if (shouldFollowLatest) {
            cancelAnimationFrame(this.scrollAnimationFrame);
            this.scrollAnimationFrame = requestAnimationFrame(() => {
                elements.transcriptScroll.scrollTo({
                    top: elements.transcriptScroll.scrollHeight,
                    behavior: addedMessage && !window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "smooth" : "auto",
                });
            });
        }
    }

    private renderMicrophone(): void {
        const started = this.conversationStarted;
        elements.sessionControls.classList.toggle("conversation-started", started);
        elements.microphoneButton.classList.toggle("active", this.microphoneOn);
        elements.microphoneButton.hidden = !started;
        elements.conversationButton.setAttribute("aria-label", started
            ? "End conversation"
            : "Start conversation");
        if (!started) {
            return;
        }
        elements.microphoneButton.setAttribute("aria-pressed", String(!this.microphoneOn));
        elements.microphoneButton.setAttribute("aria-label", this.microphoneOn ? "Mute microphone" : "Unmute microphone");
    }

    private renderTasks(): void {
        const currentSessionId = this.currentSessionId();
        const currentTasks = this.tasks.filter((task) => task.sessionId === currentSessionId);
        const currentIds = new Set(currentTasks.map((task) => task.id));
        for (const item of elements.activityList.querySelectorAll<HTMLElement>(".activity-item")) {
            if (item.dataset.sessionId !== (currentSessionId ?? "") || !currentIds.has(item.dataset.taskId!)) {
                item.remove();
            }
        }
        if (this.selectedTaskId && !currentTasks.some((task) => task.id === this.selectedTaskId)) {
            elements.taskDialog.close();
            this.closeTaskDetails();
        }
        for (const task of currentTasks) {
            const existing = Array.from(elements.activityList.querySelectorAll<HTMLElement>(".activity-item"))
                .find((item) => item.dataset.taskId === task.id);
            if (existing) {
                existing.dataset.status = task.status;
                const title = existing.querySelector("strong")!;
                if (title.textContent !== task.title) title.textContent = task.title;
                existing.querySelector<HTMLButtonElement>(".activity-whiteboard")!.hidden = !task.whiteboard;
                continue;
            }
            const item = document.createElement("article");
            item.className = "activity-item";
            item.dataset.status = task.status;
            item.dataset.taskId = task.id;
            item.dataset.sessionId = currentSessionId ?? "";
            item.innerHTML = `
                <button class="activity-row" type="button" aria-haspopup="dialog">
                    <span class="activity-copy"><strong></strong><span class="activity-status"></span></span>
                    <span class="activity-chevron" aria-hidden="true">${iconChevron()}</span>
                </button>
                <button class="activity-whiteboard" type="button" aria-label="Open task whiteboard" title="Open whiteboard">${iconWhiteboard()}</button>
            `;
            item.querySelector("strong")!.textContent = task.title;
            item.querySelector<HTMLButtonElement>(".activity-whiteboard")!.hidden = !task.whiteboard;
            item.querySelector<HTMLButtonElement>(".activity-row")!.addEventListener("click", () => {
                const latest = this.tasks.find((candidate) => candidate.id === task.id && candidate.sessionId === currentSessionId);
                if (latest) this.openTaskDetails(latest);
            });
            item.querySelector<HTMLButtonElement>(".activity-whiteboard")?.addEventListener("click", () => {
                const latest = this.tasks.find((candidate) => candidate.id === task.id && candidate.sessionId === currentSessionId);
                if (!latest?.whiteboard || !this.session) {
                    return;
                }
                this.dismissedWhiteboardTaskId = null;
                this.session.state.whiteboard = {
                    ...latest.whiteboard,
                    open: true,
                    blocks: latest.whiteboard.blocks.map((block) => ({ ...block })),
                };
            });
            elements.activityList.append(item);
        }
        elements.activityEmpty.hidden = currentTasks.length > 0;
        this.updateTaskDurations();
    }

    private openTaskDetails(task: ActivityTask): void {
        this.closeTaskDetails();
        this.selectedTaskId = task.id;
        elements.taskDialogTitle.textContent = task.title;
        elements.taskDialogAgent.textContent = `${task.backend} · Task history`;
        elements.taskMessages.textContent = "Loading activity…";
        if (!elements.taskDialog.open) {
            elements.taskDialog.show();
        }
        this.keepTaskInView();
        void this.refreshTaskDetails(task.id, this.taskDetailsRevision);
    }

    private startTaskDrag = (event: PointerEvent): void => {
        if (event.button !== 0 || !event.isPrimary || (event.target instanceof Element && event.target.closest("button"))) {
            return;
        }
        const bounds = elements.taskDialog.getBoundingClientRect();
        this.taskDrag = { pointerId: event.pointerId, offsetX: event.clientX - bounds.left, offsetY: event.clientY - bounds.top };
        elements.taskDialogHeader.setPointerCapture(event.pointerId);
        elements.taskDialog.classList.add("is-dragging");
        event.preventDefault();
    };

    private moveTaskDrag = (event: PointerEvent): void => {
        if (!this.taskDrag || event.pointerId !== this.taskDrag.pointerId) return;
        elements.taskDialog.style.left = `${event.clientX - this.taskDrag.offsetX}px`;
        elements.taskDialog.style.top = `${event.clientY - this.taskDrag.offsetY}px`;
        elements.taskDialog.style.right = "auto";
        this.keepTaskInView();
    };

    private endTaskDrag = (event: PointerEvent): void => {
        if (!this.taskDrag || event.pointerId !== this.taskDrag.pointerId) return;
        this.taskDrag = null;
        elements.taskDialog.classList.remove("is-dragging");
        if (elements.taskDialogHeader.hasPointerCapture(event.pointerId)) {
            elements.taskDialogHeader.releasePointerCapture(event.pointerId);
        }
    };

    private keepTaskInView = (): void => {
        if (!elements.taskDialog.open) return;
        const bounds = elements.taskDialog.getBoundingClientRect();
        const left = Math.max(8, Math.min(bounds.left, window.innerWidth - bounds.width - 8));
        const top = Math.max(8, Math.min(bounds.top, window.innerHeight - bounds.height - 8));
        elements.taskDialog.style.left = `${left}px`;
        elements.taskDialog.style.top = `${top}px`;
        elements.taskDialog.style.right = "auto";
    };

    private closeTaskDetails = (): void => {
        this.selectedTaskId = null;
        this.taskDetailsRevision += 1;
        this.taskMessagesSignature = "";
        window.clearTimeout(this.taskDetailsTimer);
    };

    private async refreshTaskDetails(taskId: string, revision: number): Promise<void> {
        try {
            const messages = await this.session!.getTaskMessages(taskId);
            if (revision !== this.taskDetailsRevision || this.selectedTaskId !== taskId) {
                return;
            }
            const signature = JSON.stringify(messages);
            if (signature !== this.taskMessagesSignature) {
                const nearBottom = elements.taskMessages.scrollHeight - elements.taskMessages.scrollTop
                    - elements.taskMessages.clientHeight < 60;
                this.taskMessagesSignature = signature;
                elements.taskMessages.replaceChildren();
                for (const message of messages) {
                    const row = document.createElement("article");
                    row.className = "task-message";
                    row.dataset.role = message.role;
                    const label = document.createElement("span");
                    label.className = "task-message-label";
                    const task = this.tasks.find((candidate) => candidate.id === taskId);
                    label.textContent = message.role === "user" ? "Task request"
                        : message.role === "tool" ? "Tool" : task?.backend ?? "Agent";
                    const time = document.createElement("time");
                    time.dateTime = new Date(message.created_at * 1000).toISOString();
                    time.textContent = new Date(message.created_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                    const heading = document.createElement("div");
                    heading.className = "task-message-heading";
                    heading.append(label, time);
                    const content = document.createElement("div");
                    content.className = "task-message-content";
                    const sections = message.content.split(/(```[\s\S]*?```)/g);
                    for (const section of sections) {
                        if (!section) continue;
                        if (section.startsWith("```") && section.endsWith("```")) {
                            const code = document.createElement("code");
                            code.textContent = section.slice(3, -3).replace(/^[\w+-]*\n/, "");
                            const pre = document.createElement("pre");
                            pre.append(code);
                            content.append(pre);
                        } else {
                            const paragraph = document.createElement("p");
                            paragraph.textContent = section;
                            content.append(paragraph);
                        }
                    }
                    row.append(heading, content);
                    elements.taskMessages.append(row);
                }
                if (!messages.length) {
                    elements.taskMessages.textContent = "No activity yet.";
                }
                this.keepTaskInView();
                if (nearBottom) {
                    elements.taskMessages.scrollTop = elements.taskMessages.scrollHeight;
                }
            }
        } catch {
            if (revision === this.taskDetailsRevision) {
                elements.taskMessages.textContent = "Activity is unavailable. Retrying…";
            }
        }
        if (revision === this.taskDetailsRevision && this.selectedTaskId === taskId) {
            this.taskDetailsTimer = window.setTimeout(() => void this.refreshTaskDetails(taskId, revision), 1500);
        }
    }

    private renderToolActivity(
        toolCall: SessionState["tool_call"],
        tasks: SessionState["tasks"],
    ): void {
        const name = toolCall.name;
        if (!["delegate", "update", "inspect", "cancel"].includes(name)) {
            this.lastToolSignature = "";
            elements.toolActivity.hidden = true;
            return;
        }
        const args = toolCall.args;
        const taskId = typeof args.task_id === "string" ? args.task_id : "";
        const taskTitle = tasks.find((task) => task.id === taskId)?.title;
        let detail = taskTitle || "当前任务";
        if (name === "delegate") {
            detail = String(args.task ?? "");
        } else if (name === "update") {
            detail = String(args.instruction ?? "");
        }
        const signature = `${name}:${taskId}:${detail}:${toolCall.calledAt ?? ""}`;
        if (signature === this.lastToolSignature) {
            return;
        }
        this.lastToolSignature = signature;
        const compact = detail.replace(/\s+/g, " ").trim();
        const visible = compact.length > 46 ? `${compact.slice(0, 45)}…` : compact;
        elements.toolActivity.textContent = visible ? `${name}「${visible}」` : name;
        elements.toolActivity.hidden = false;
        elements.toolActivity.classList.remove("tool-activity-visible");
        void elements.toolActivity.offsetWidth;
        elements.toolActivity.classList.add("tool-activity-visible");
        window.clearTimeout(this.toolActivityTimer);
        this.toolActivityTimer = window.setTimeout(() => {
            elements.toolActivity.hidden = true;
            elements.toolActivity.classList.remove("tool-activity-visible");
        }, 5200);
    }

    private renderWhiteboard(whiteboard: SessionState["whiteboard"]): void {
        if (
            !whiteboard.open
            || !whiteboard.taskId
            || whiteboard.taskId === this.dismissedWhiteboardTaskId
        ) {
            elements.whiteboardWindow.hidden = true;
            return;
        }
        if (this.dismissedWhiteboardTaskId && this.dismissedWhiteboardTaskId !== whiteboard.taskId) {
            this.dismissedWhiteboardTaskId = null;
        }
        elements.whiteboardTitle.textContent = whiteboard.title || "Whiteboard";
        elements.whiteboardContent.replaceChildren();
        let newestBlock: HTMLElement | null = null;
        for (const [index, block] of whiteboard.blocks.entries()) {
            const item = document.createElement("div");
            item.className = `whiteboard-block whiteboard-block-${block.kind}`;
            item.dataset.blockId = block.id;
            item.classList.toggle("whiteboard-block-highlighted", block.highlighted);
            item.classList.toggle("whiteboard-block-current", index === whiteboard.blocks.length - 1);
            if (block.kind === "equation") {
                renderEquation(item, block.content);
            } else {
                item.textContent = block.content;
            }
            const signature = `${whiteboard.taskId}:${block.id}`;
            if (!this.renderedWhiteboardBlocks.has(signature)) {
                item.classList.add("whiteboard-block-entering");
                this.renderedWhiteboardBlocks.add(signature);
                newestBlock = item;
            }
            elements.whiteboardContent.append(item);
        }
        elements.whiteboardEmpty.hidden = whiteboard.blocks.length > 0;
        elements.whiteboardWindow.hidden = false;
        if (newestBlock) {
            window.requestAnimationFrame(() => {
                newestBlock?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            });
        }
    }

    private hideWhiteboard = async (): Promise<void> => {
        const taskId = this.session?.state.whiteboard.taskId ?? null;
        this.dismissedWhiteboardTaskId = taskId;
        elements.whiteboardWindow.hidden = true;
        const task = this.tasks.find((item) => item.id === taskId);
        if (taskId && task?.status === "running" && this.session) {
            try {
                await this.session.cancelTask(taskId);
            } catch (error) {
                this.dismissedWhiteboardTaskId = null;
                this.renderWhiteboard(this.session.state.whiteboard);
                this.updateStatus(readableConnectionError(error));
            }
        }
    };

    private startWhiteboardDrag = (event: PointerEvent): void => {
        if (
            event.button !== 0
            || (event.target as Element | null)?.closest("button")
        ) {
            return;
        }
        const bounds = elements.whiteboardWindow.getBoundingClientRect();
        this.whiteboardDrag = {
            pointerId: event.pointerId,
            offsetX: event.clientX - bounds.left,
            offsetY: event.clientY - bounds.top,
        };
        elements.whiteboardHeader.setPointerCapture(event.pointerId);
        elements.whiteboardWindow.classList.add("whiteboard-dragging");
        event.preventDefault();
    };

    private moveWhiteboard = (event: PointerEvent): void => {
        const drag = this.whiteboardDrag;
        if (!drag || drag.pointerId !== event.pointerId) {
            return;
        }
        const width = elements.whiteboardWindow.offsetWidth;
        const height = elements.whiteboardWindow.offsetHeight;
        const left = Math.min(
            Math.max(12, event.clientX - drag.offsetX),
            Math.max(12, window.innerWidth - width - 12),
        );
        const top = Math.min(
            Math.max(72, event.clientY - drag.offsetY),
            Math.max(72, window.innerHeight - height - 12),
        );
        elements.whiteboardWindow.style.left = `${left}px`;
        elements.whiteboardWindow.style.top = `${top}px`;
        elements.whiteboardWindow.style.right = "auto";
    };

    private endWhiteboardDrag = (event: PointerEvent): void => {
        if (!this.whiteboardDrag || this.whiteboardDrag.pointerId !== event.pointerId) {
            return;
        }
        if (elements.whiteboardHeader.hasPointerCapture(event.pointerId)) {
            elements.whiteboardHeader.releasePointerCapture(event.pointerId);
        }
        this.whiteboardDrag = null;
        elements.whiteboardWindow.classList.remove("whiteboard-dragging");
    };

    private refreshSessionHistory = async (session: Session): Promise<void> => {
        this.sessionHistoryState = "loading";
        this.renderSessionHistory();
        try {
            const sessions = await session.getSessions();
            if (this.session !== session || this.destroyed) {
                return;
            }
            this.sessionHistory = sessions;
            this.sessionHistoryState = "ready";
        } catch {
            if (this.session !== session || this.destroyed) {
                return;
            }
            this.sessionHistory = [];
            this.sessionHistoryState = "error";
        }
        this.renderSessionHistory();
    };

    private renderSessionHistory(): void {
        elements.sessionHistoryList.replaceChildren();
        const currentSessionId = this.currentSessionId();
        for (const summary of this.sessionHistory) {
            const row = document.createElement("div");
            row.className = "session-history-row";
            row.dataset.sessionId = summary.session_id;
            const button = document.createElement("button");
            button.className = "session-history-item";
            button.type = "button";
            button.dataset.sessionId = summary.session_id;
            const current = summary.session_id === currentSessionId;
            if (current) {
                button.setAttribute("aria-current", "true");
                row.classList.add("current");
            }

            const title = document.createElement("strong");
            title.textContent = summary.title?.trim() || "Untitled session";
            const time = document.createElement("time");
            const formattedTime = formatSessionTime(summary.updated_at);
            time.textContent = formattedTime;
            time.hidden = !formattedTime;
            if (summary.updated_at) {
                time.dateTime = summary.updated_at;
            }
            button.append(title, time);

            const renameButton = document.createElement("button");
            renameButton.className = "session-rename-button";
            renameButton.type = "button";
            renameButton.dataset.renameSessionId = summary.session_id;
            renameButton.setAttribute("aria-label", `Rename ${title.textContent}`);
            renameButton.innerHTML = iconRename();

            const deleteButton = document.createElement("button");
            deleteButton.className = "session-delete-button";
            deleteButton.type = "button";
            deleteButton.dataset.deleteSessionId = summary.session_id;
            deleteButton.setAttribute("aria-label", `Delete ${title.textContent}`);
            deleteButton.innerHTML = iconDelete();
            row.append(button, renameButton, deleteButton);
            elements.sessionHistoryList.append(row);
        }

        elements.sessionHistoryEmpty.hidden = this.sessionHistory.length > 0;
        elements.sessionHistoryEmpty.textContent = this.sessionHistoryState === "loading"
            ? "Loading session history…"
            : this.sessionHistoryState === "error"
                ? "Session history is unavailable."
                : this.sessionHistoryState === "ready"
                    ? "No saved sessions yet."
                    : "Start a conversation to load session history.";
    }

    private handleSessionHistoryClick = async (event: MouseEvent): Promise<void> => {
        const deleteButton = (event.target as Element | null)?.closest<HTMLButtonElement>("button[data-delete-session-id]");
        if (deleteButton?.dataset.deleteSessionId) {
            await this.deleteSession(deleteButton.dataset.deleteSessionId, deleteButton);
            return;
        }
        const renameButton = (event.target as Element | null)?.closest<HTMLButtonElement>("button[data-rename-session-id]");
        if (renameButton?.dataset.renameSessionId) {
            this.beginSessionRename(renameButton.dataset.renameSessionId);
            return;
        }
        const button = (event.target as Element | null)?.closest<HTMLButtonElement>("button[data-session-id]");
        const session = this.session;
        const sessionId = button?.dataset.sessionId;
        if (!button || !sessionId || button.disabled) {
            return;
        }
        this.closeSessionMenu();
        if (!session) {
            return;
        }

        button.disabled = true;
        this.updateStatus("Opening session");
        try {
            await session.switchSession(sessionId);
            await session.open();
            this.microphoneOn = !session.muted;
            this.conversationStarted = true;
            this.setConnectionState("connected");
            this.renderMicrophone();
            this.renderSessionHistory();
            this.updateStatus("Ready to listen");
        } catch (error) {
            this.setConnectionState("error");
            this.updateStatus(readableConnectionError(error));
        } finally {
            button.disabled = false;
        }
    };

    private deleteSession = async (
        sessionId: string,
        button: HTMLButtonElement,
    ): Promise<void> => {
        const summary = this.sessionHistory.find((item) => item.session_id === sessionId);
        if (!summary) {
            return;
        }
        const title = summary.title?.trim() || "Untitled session";
        if (!window.confirm(`Delete “${title}”? This cannot be undone.`)) {
            return;
        }

        const deletingCurrent = sessionId === this.currentSessionId();
        button.disabled = true;
        this.cancelSessionRename();
        try {
            const session = this.session;
            if (!session) {
                throw new Error("No live session is connected");
            }
            await session.deleteSession(sessionId);
            this.sessionHistory = this.sessionHistory.filter(
                (item) => item.session_id !== sessionId
            );
            if (deletingCurrent) {
                this.microphoneOn = false;
                this.conversationStarted = false;
                this.inputLevel = 0;
                this.outputLevel = 0;
                this.tasks = [];
                this.renderMessages([], "live:new");
                this.renderTasks();
                this.orb.setState({
                    audioLevel: 0,
                    speaker: "idle",
                    connectionState: "disconnected",
                });
                this.renderMicrophone();
                this.updateStatus("Session deleted");
            }
            this.renderSessionHistory();
        } catch (error) {
            button.disabled = false;
            this.updateStatus(readableConnectionError(error));
        }
    };

    private beginSessionRename(sessionId: string): void {
        const summary = this.sessionHistory.find((session) => session.session_id === sessionId);
        const row = elements.sessionHistoryList.querySelector<HTMLElement>(`.session-history-row[data-session-id="${CSS.escape(sessionId)}"]`);
        if (!summary || !row || row.classList.contains("renaming")) {
            return;
        }
        this.cancelSessionRename();
        row.classList.add("renaming");
        const form = document.createElement("form");
        form.className = "session-rename-form";
        form.dataset.sessionId = sessionId;
        const input = document.createElement("input");
        input.className = "session-rename-input";
        input.name = "title";
        input.maxLength = 80;
        input.value = summary.title?.trim() || "";
        input.setAttribute("aria-label", "Session name");
        const save = document.createElement("button");
        save.className = "session-rename-save";
        save.type = "submit";
        save.setAttribute("aria-label", "Save session name");
        save.innerHTML = iconCheck();
        form.append(input, save);
        row.append(form);
        input.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key === "Escape") {
                keyEvent.stopPropagation();
                this.cancelSessionRename();
            }
        });
        input.focus();
        input.select();
    }

    private handleSessionRenameSubmit = async (event: SubmitEvent): Promise<void> => {
        const form = event.target instanceof HTMLFormElement ? event.target : null;
        const sessionId = form?.dataset.sessionId;
        const input = form?.querySelector<HTMLInputElement>("input[name=title]");
        const title = input?.value.trim() ?? "";
        if (!form || !sessionId || !input || !title) {
            event.preventDefault();
            input?.focus();
            return;
        }
        event.preventDefault();
        const controls = form.querySelectorAll<HTMLButtonElement | HTMLInputElement>("button, input");
        controls.forEach((control) => { control.disabled = true; });
        try {
            let updated: SessionSummary;
            const session = this.session;
            if (!session) {
                throw new Error("No live session is connected");
            }
            updated = await session.renameSession(sessionId, title);
            this.sessionHistory = this.sessionHistory.map((summary) =>
                summary.session_id === sessionId ? { ...summary, ...updated } : summary
            );
            this.renderSessionHistory();
        } catch (error) {
            controls.forEach((control) => { control.disabled = false; });
            input.setCustomValidity(readableConnectionError(error));
            input.reportValidity();
            input.addEventListener("input", () => input.setCustomValidity(""), { once: true });
            input.focus();
        }
    };

    private cancelSessionRename(): void {
        for (const row of elements.sessionHistoryList.querySelectorAll(".session-history-row.renaming")) {
            row.classList.remove("renaming");
            row.querySelector(".session-rename-form")?.remove();
        }
    }

    private updateTaskDurations = (): void => {
        const currentSessionId = this.currentSessionId();
        for (const element of elements.activityList.querySelectorAll<HTMLElement>(".activity-item[data-task-id]")) {
            const task = this.tasks.find((candidate) =>
                candidate.id === element.dataset.taskId && candidate.sessionId === currentSessionId
            );
            const status = element.querySelector<HTMLElement>(".activity-status");
            if (!task || !status) {
                continue;
            }
            const label = task.status === "running"
                ? "In progress"
                : task.status === "completed" ? "Completed" : task.status;
            status.textContent = `${label} · ${formatTaskDuration(task.startedAt, task.completedAt)}`;
        }
    };

    private setConnectionState(state: OrbConnectionState): void {
        this.orb.setState({ connectionState: state });
    }

    private toggleSessionMenu = (): void => {
        const open = elements.sessionSwitcher.dataset.open !== "true";
        elements.sessionSwitcher.dataset.open = String(open);
        elements.sessionMenuButton.setAttribute("aria-expanded", String(open));
        elements.sessionPopover.setAttribute("aria-hidden", String(!open));
    };

    private closeSessionMenu(): void {
        this.cancelSessionRename();
        elements.sessionSwitcher.dataset.open = "false";
        elements.sessionMenuButton.setAttribute("aria-expanded", "false");
        elements.sessionPopover.setAttribute("aria-hidden", "true");
    }

    private handleDocumentPointerDown = (event: PointerEvent): void => {
        if (event.target instanceof Node && !elements.sessionSwitcher.contains(event.target)) {
            this.closeSessionMenu();
        }
    };

    private handleDocumentKeyDown = (event: KeyboardEvent): void => {
        if (event.key === "Escape") {
            if (elements.taskDialog.open) {
                elements.taskDialog.close();
            } else {
                this.closeSessionMenu();
                elements.sessionMenuButton.focus();
            }
        }
    };

    private currentSessionId(): string | null {
        return this.session?.state.sessionId ?? null;
    }

    private endSession = async (): Promise<void> => {
        const session = this.session;
        if (session) {
            await session.close().catch(() => undefined);
            await session.switchSession(null).catch(() => undefined);
        }
        this.microphoneOn = false;
        this.conversationStarted = false;
        this.inputLevel = 0;
        this.outputLevel = 0;
        this.tasks = [];
        this.renderMessages([], "live:new");
        this.renderTasks();
        this.orb.setState({ audioLevel: 0, speaker: "idle", connectionState: "disconnected" });
        this.renderMicrophone();
        this.updateStatus("Session ended");
        if (session) {
            void this.refreshSessionHistory(session);
        }
    };

    private closeLiveSession = async (): Promise<void> => {
        const session = this.session;
        this.session = null;
        this.inputMeter.stop();
        this.outputMeter.stop();
        this.tasks = [];
        this.renderTasks();
        this.renderSessionHistory();
        if (session) {
            await session.close().catch(() => undefined);
        }
    }

    private dispose = (): void => {
        if (this.destroyed) {
            return;
        }
        this.destroyed = true;
        this.closeTaskDetails();
        window.clearInterval(this.taskClockTimer);
        window.clearTimeout(this.toolActivityTimer);
        elements.taskDialogHeader.removeEventListener("pointerdown", this.startTaskDrag);
        elements.taskDialogHeader.removeEventListener("pointermove", this.moveTaskDrag);
        elements.taskDialogHeader.removeEventListener("pointerup", this.endTaskDrag);
        elements.taskDialogHeader.removeEventListener("pointercancel", this.endTaskDrag);
        elements.taskDialogHeader.removeEventListener("lostpointercapture", this.endTaskDrag);
        window.removeEventListener("resize", this.keepTaskInView);
        elements.voiceToggle.removeEventListener("click", this.changeVoice);
        elements.voiceToggle.removeEventListener("pointerdown", this.startVoiceDrag);
        elements.voiceToggle.removeEventListener("pointermove", this.moveVoiceDrag);
        elements.voiceToggle.removeEventListener("pointerup", this.endVoiceDrag);
        elements.voiceToggle.removeEventListener("pointercancel", this.cancelVoiceDrag);
        elements.voiceToggle.removeEventListener("lostpointercapture", this.cancelVoiceDrag);
        elements.whiteboardHide.removeEventListener("click", this.hideWhiteboard);
        elements.whiteboardHeader.removeEventListener("pointerdown", this.startWhiteboardDrag);
        window.removeEventListener("pointermove", this.moveWhiteboard);
        window.removeEventListener("pointerup", this.endWhiteboardDrag);
        window.removeEventListener("pointercancel", this.endWhiteboardDrag);
        document.removeEventListener("pointerdown", this.handleDocumentPointerDown);
        document.removeEventListener("keydown", this.handleDocumentKeyDown);
        cancelAnimationFrame(this.scrollAnimationFrame);
        for (const timer of this.messageExitTimers.values()) {
            window.clearTimeout(timer);
        }
        this.messageExitTimers.clear();
        void this.closeLiveSession();
        this.orb.dispose();
    };
}

new XLiveApp();

function iconClose(): string {
    return `
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6.5 6.5l11 11M17.5 6.5l-11 11"></path>
        </svg>
    `;
}

function requireElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`Missing element: ${id}`);
    }
    return element;
}

function requireButton(id: string): HTMLButtonElement {
    return requireElement(id) as HTMLButtonElement;
}

function requireAnchor(id: string): HTMLAnchorElement {
    return requireElement(id) as HTMLAnchorElement;
}

function isWebSocketURL(value: string): boolean {
    try {
        const url = new URL(value);
        return url.protocol === "ws:" || url.protocol === "wss:";
    } catch {
        return false;
    }
}

function readableConnectionError(error: unknown): string {
    if (error instanceof Error && error.message) {
        return `Could not connect: ${error.message}`;
    }
    return "Could not connect to the X-Live runtime.";
}

function formatSessionTime(value: string | null | undefined): string {
    if (!value) {
        return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return "";
    }
    const now = new Date();
    const sameDay = date.getFullYear() === now.getFullYear()
        && date.getMonth() === now.getMonth()
        && date.getDate() === now.getDate();
    const time = new Intl.DateTimeFormat(undefined, {
        hour: "2-digit",
        minute: "2-digit",
    }).format(date);
    if (sameDay) {
        return `Today · ${time}`;
    }
    const day = new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
    }).format(date);
    return `${day} · ${time}`;
}

function formatTaskDuration(startedAt: string, completedAt?: string | null): string {
    const started = new Date(startedAt).getTime();
    const ended = completedAt ? new Date(completedAt).getTime() : Date.now();
    if (!Number.isFinite(started) || !Number.isFinite(ended)) {
        return "--:--";
    }
    const totalSeconds = Math.max(0, Math.floor((ended - started) / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
        ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
        : `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function iconMicrophone(): string {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15.4a3.4 3.4 0 0 0 3.4-3.4V7a3.4 3.4 0 1 0-6.8 0v5a3.4 3.4 0 0 0 3.4 3.4Z"/><path d="M18.5 11.5v.5a6.5 6.5 0 0 1-13 0v-.5M12 18.5V22M8.8 22h6.4"/></svg>`;
}

function iconMicrophoneOff(): string {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 16M9 9v3a3 3 0 0 0 4.7 2.5M15 11.2V7a3 3 0 0 0-5.8-1M18.5 11.5v.5a6.5 6.5 0 0 1-10.7 5M5.5 11.5v.5a6.5 6.5 0 0 0 9.7 5.7M12 18.5V22M8.8 22h6.4"/></svg>`;
}

function iconEnd(): string {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 7 10 10M17 7 7 17"/></svg>`;
}

function iconGitHub(): string {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.8a9.4 9.4 0 0 0-3 18.3c.47.08.64-.2.64-.45v-1.8c-2.62.57-3.17-1.1-3.17-1.1-.43-1.1-1.05-1.39-1.05-1.39-.86-.58.06-.57.06-.57.95.07 1.45.98 1.45.98.84 1.45 2.21 1.03 2.75.79.08-.61.33-1.03.6-1.27-2.09-.24-4.29-1.05-4.29-4.65 0-1.03.37-1.87.97-2.53-.1-.24-.42-1.2.09-2.5 0 0 .79-.25 2.58.97A8.98 8.98 0 0 1 12 7.25c.8 0 1.6.1 2.35.32 1.79-1.22 2.57-.97 2.57-.97.51 1.3.19 2.26.1 2.5.6.66.96 1.5.96 2.53 0 3.61-2.2 4.4-4.3 4.64.34.3.64.87.64 1.76v2.62c0 .25.17.54.65.45A9.4 9.4 0 0 0 12 2.8Z"/></svg>`;
}

function iconDemo(): string {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="4.5" width="17" height="14" rx="2"/><path d="M3.5 8h17M10 11l4 2.25-4 2.25z"/></svg>`;
}

function iconWhiteboard(): string {
    return `<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3.5" y="4" width="13" height="10" rx="1.5"/><path d="M7 17h6M10 14v3M6.5 8h7M6.5 10.8h4.5"/></svg>`;
}

function renderEquation(element: HTMLElement, source: string): void {
    const expression = source.trim()
        .replace(/^\$\$|\$\$$/g, "")
        .replace(/^\$|\$$/g, "")
        .replace(/^\\\[|\\\]$/g, "")
        .replace(/^\\\(|\\\)$/g, "")
        .trim();
    try {
        katex.render(expression, element, {
            displayMode: true,
            throwOnError: false,
            strict: "ignore",
            trust: false,
        });
        element.setAttribute("aria-label", source);
    } catch {
        element.textContent = source;
    }
}

function iconChevron(): string {
    return `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m6.5 8 3.5 3.5L13.5 8"/></svg>`;
}

function iconRename(): string {
    return `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m12.8 4.2 3 3-7.9 7.9-3.6.7.7-3.6zM11.7 5.3l3 3"/></svg>`;
}

function iconDelete(): string {
    return `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.8 6.2h10.4M8 3.8h4l.7 2.4H7.3zM6.2 6.2l.7 9h6.2l.7-9M8.5 8.7v4M11.5 8.7v4"/></svg>`;
}

function iconCheck(): string {
    return `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5.5 10.2 2.8 2.8 6.2-6.2"/></svg>`;
}
