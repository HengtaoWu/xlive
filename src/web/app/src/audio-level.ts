export type AudioLevelListener = (level: number) => void;

export class AudioLevelSmoother {
    private value = 0;
    private noiseFloor = 0.012;
    private peak = 0.12;
    private lastTimestamp = 0;

    update(rawRms: number, timestamp = performance.now()): number {
        // Browser microphone processing can leave ordinary speech surprisingly quiet.
        // A small input gain keeps the gate useful without mapping raw peaks directly.
        const safeRms = Number.isFinite(rawRms) ? Math.max(0, rawRms) * 1.35 : 0;
        const deltaSeconds = this.lastTimestamp > 0
            ? Math.min(0.1, (timestamp - this.lastTimestamp) / 1000)
            : 1 / 60;
        this.lastTimestamp = timestamp;

        if (safeRms < 0.055) {
            this.noiseFloor += (safeRms - this.noiseFloor) * 0.006;
        } else {
            this.noiseFloor *= 0.9999;
        }
        this.noiseFloor = clamp(this.noiseFloor, 0.006, 0.026);

        this.peak = Math.max(safeRms, this.peak * Math.pow(0.996, deltaSeconds * 60));
        const gate = clamp(this.noiseFloor + 0.018, 0.026, 0.046);
        this.peak = Math.max(this.peak, gate + 0.07);

        const gateBlend = smoothstep(gate, gate + 0.01, safeRms);
        const gated = Math.max(0, safeRms - gate);
        const range = Math.max(0.07, this.peak - gate);
        const normalized = Math.pow(clamp(gated / range, 0, 0.82), 0.82) * gateBlend;
        const timeConstant = normalized > this.value ? 0.125 : 0.60;
        const coefficient = 1 - Math.exp(-deltaSeconds / timeConstant);
        this.value += (normalized - this.value) * coefficient;

        if (this.value < 0.004) {
            this.value = 0;
        }
        return clamp(this.value, 0, 0.82);
    }

    reset(): void {
        this.value = 0;
        this.lastTimestamp = 0;
    }
}

export class PCMLevelMeter {
    private readonly listener: AudioLevelListener;
    private readonly smoother = new AudioLevelSmoother();
    private targetRms = 0;
    private animationFrame = 0;
    private running = false;

    constructor(listener: AudioLevelListener) {
        this.listener = listener;
    }

    pushPCM16(buffer: ArrayBuffer): void {
        const samples = new Int16Array(buffer);
        if (!samples.length) {
            return;
        }
        let sumSquares = 0;
        for (let index = 0; index < samples.length; index += 1) {
            const sample = (samples[index] ?? 0) / 32768;
            sumSquares += sample * sample;
        }
        this.targetRms = Math.sqrt(sumSquares / samples.length);
        if (!this.running) {
            this.running = true;
            this.tick();
        }
    }

    stop(): void {
        this.running = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = 0;
        }
        this.targetRms = 0;
        this.smoother.reset();
        this.listener(0);
    }

    private tick = (): void => {
        if (!this.running) {
            return;
        }
        this.listener(this.smoother.update(this.targetRms));
        this.targetRms *= 0.975;
        this.animationFrame = requestAnimationFrame(this.tick);
    };
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}

function smoothstep(edge0: number, edge1: number, value: number): number {
    const normalized = clamp((value - edge0) / (edge1 - edge0), 0, 1);
    return normalized * normalized * (3 - 2 * normalized);
}
