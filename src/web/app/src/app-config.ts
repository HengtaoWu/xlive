export interface XLiveAppConfig {
    websocketURL: string;
    githubURL: string;
}

declare global {
    interface Window {
        /**
         * Optional deployment-time overrides. Define this before the app bundle
         * loads; secrets must never be placed here because it is browser-visible.
         */
        __XLIVE_CONFIG__?: Partial<XLiveAppConfig>;
    }
}

const DEFAULT_GITHUB_URL = "https://github.com/HengtaoWu/xlive";

export function resolveAppConfig(): XLiveAppConfig {
    const configured = window.__XLIVE_CONFIG__ ?? {};
    return {
        websocketURL: configured.websocketURL ?? defaultWebSocketURL(),
        githubURL: configured.githubURL ?? DEFAULT_GITHUB_URL,
    };
}

function defaultWebSocketURL(): string {
    const url = new URL("./xlive-ws", window.location.href);
    url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    if ((url.hostname === "127.0.0.1" || url.hostname === "localhost") && url.port === "8000") {
        url.port = "8001";
        url.pathname = "/ws";
    }
    return url.toString();
}
