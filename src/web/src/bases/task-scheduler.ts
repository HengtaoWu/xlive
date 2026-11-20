export { BaseDeferredTaskScheduler };

abstract class BaseDeferredTaskScheduler {
    abstract schedule(task: () => void): void;

    abstract dispose(): void;
}
