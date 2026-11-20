import { BaseDeferredTaskScheduler } from "./bases/task-scheduler";
import { getPlatformRuntime } from "./platforms/index";

export { createDeferredTaskScheduler };

function createDeferredTaskScheduler(): BaseDeferredTaskScheduler {
    return getPlatformRuntime().createDeferredTaskScheduler();
}
