// Inspired by Cammy, in both looks and voice.
import {
    FrontSide,
    Group,
    Mesh,
    NormalBlending,
    PerspectiveCamera,
    Scene,
    ShaderMaterial,
    SphereGeometry,
    SRGBColorSpace,
    WebGLRenderer,
} from "three";

export type OrbSpeaker = "idle" | "user" | "agent";
export type OrbConnectionState =
    | "connecting"
    | "connected"
    | "reconnecting"
    | "disconnected"
    | "error";

export interface AudioReactiveOrbState {
    audioLevel: number;
    speaker: OrbSpeaker;
    connectionState: OrbConnectionState;
}

const sharedNoise = /* glsl */ `
    float hash31(vec3 p) {
        p = fract(p * 0.1031);
        p += dot(p, p.yzx + 33.33);
        return fract((p.x + p.y) * p.z);
    }

    float valueNoise(vec3 p) {
        vec3 cell = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
            mix(mix(hash31(cell), hash31(cell + vec3(1.0, 0.0, 0.0)), f.x),
                mix(hash31(cell + vec3(0.0, 1.0, 0.0)), hash31(cell + vec3(1.0, 1.0, 0.0)), f.x), f.y),
            mix(mix(hash31(cell + vec3(0.0, 0.0, 1.0)), hash31(cell + vec3(1.0, 0.0, 1.0)), f.x),
                mix(hash31(cell + vec3(0.0, 1.0, 1.0)), hash31(cell + vec3(1.0, 1.0, 1.0)), f.x), f.y),
            f.z
        );
    }

    float fbm(vec3 p) {
        float value = 0.0;
        float amplitude = 0.52;
        for (int octave = 0; octave < 4; octave++) {
            value += amplitude * valueNoise(p);
            p = p * 2.02 + vec3(1.7, 3.1, 2.4);
            amplitude *= 0.5;
        }
        return value;
    }
`;

const shellVertexShader = /* glsl */ `
    uniform float uTime;
    uniform float uLevel;
    uniform float uMotion;

    varying vec3 vLocalPosition;
    varying vec3 vNormalView;
    varying vec3 vViewDirection;

    ${sharedNoise}

    void main() {
        float drift = fbm(position * 1.18 + vec3(uTime * 0.018, -uTime * 0.012, uTime * 0.009));
        float displacement = (drift - 0.5) * (0.009 + uLevel * 0.043) * uMotion;
        vec3 displaced = position + normal * displacement;
        vLocalPosition = displaced;
        vec4 viewPosition = modelViewMatrix * vec4(displaced, 1.0);
        vNormalView = normalize(normalMatrix * normal);
        vViewDirection = normalize(-viewPosition.xyz);
        gl_Position = projectionMatrix * viewPosition;
    }
`;

const shellFragmentShader = /* glsl */ `
    precision highp float;

    uniform float uLevel;
    uniform float uSpeaker;
    uniform float uConnected;

    varying vec3 vLocalPosition;
    varying vec3 vNormalView;
    varying vec3 vViewDirection;

    void main() {
        vec3 normalView = normalize(vNormalView);
        float facing = max(dot(normalView, normalize(vViewDirection)), 0.0);
        float edge = pow(1.0 - facing, 2.88);
        float softEdge = pow(1.0 - facing, 1.02);
        float key = pow(max(dot(normalView, normalize(vec3(-0.52, 0.68, 0.94))), 0.0), 28.0);
        float fill = pow(max(dot(normalView, normalize(vec3(0.64, -0.36, 0.76))), 0.0), 11.0);
        float topWash = smoothstep(-0.9, 0.92, vLocalPosition.y);
        float rimWeight = mix(0.74, 1.0, smoothstep(-0.92, 0.72, vLocalPosition.y));
        rimWeight *= mix(0.90, 1.0, smoothstep(-0.88, 0.82, vLocalPosition.x));

        vec3 ice = vec3(0.80, 0.91, 1.0);
        vec3 lilac = vec3(0.89, 0.86, 1.0);
        vec3 activeTint = mix(vec3(0.70, 0.86, 1.0), lilac, uSpeaker);
        vec3 color = mix(ice, lilac, clamp(vLocalPosition.x * 0.18 + 0.48, 0.0, 1.0));
        color = mix(color, vec3(0.99, 1.0, 1.0), 0.38 + topWash * 0.18);
        color = mix(color, activeTint, uLevel * 0.075 + uConnected * 0.025);
        color += vec3(1.0) * key * 0.76;
        color += vec3(0.72, 0.82, 1.0) * fill * 0.10;
        vec3 rimTint = mix(vec3(0.70, 0.84, 1.0), lilac, 0.38);
        color += rimTint * edge * rimWeight * (0.23 + uLevel * 0.10);

        float alpha = 0.054 + softEdge * 0.092 + edge * rimWeight * 0.35;
        alpha += key * 0.29 + fill * 0.045 + uLevel * 0.034;
        gl_FragColor = vec4(color, clamp(alpha, 0.0, 0.86));
    }
`;

const liquidVertexShader = /* glsl */ `
    uniform float uTime;
    uniform float uLevel;
    uniform float uMotion;

    varying vec3 vLocalPosition;
    varying vec3 vNormalView;
    varying vec3 vViewDirection;

    ${sharedNoise}

    void main() {
        float slowNoise = fbm(position * 1.22 + vec3(uTime * 0.014, -uTime * 0.010, 0.0));
        float displacement = (slowNoise - 0.5) * (0.007 + uLevel * 0.035) * uMotion;
        vec3 displaced = position + normal * displacement;
        vLocalPosition = displaced;
        vec4 viewPosition = modelViewMatrix * vec4(displaced, 1.0);
        vNormalView = normalize(normalMatrix * normal);
        vViewDirection = normalize(-viewPosition.xyz);
        gl_Position = projectionMatrix * viewPosition;
    }
`;

const liquidFragmentShader = /* glsl */ `
    precision highp float;

    uniform float uTime;
    uniform float uLevel;
    uniform float uMotion;
    uniform float uSpeaker;

    varying vec3 vLocalPosition;
    varying vec3 vNormalView;
    varying vec3 vViewDirection;

    ${sharedNoise}

    void main() {
        vec3 p = vLocalPosition;
        float flowTime = uTime * uMotion;
        float broadNoise = fbm(p * 1.05 + vec3(flowTime * 0.30, -flowTime * 0.22, flowTime * 0.16));

        float twist = -0.27 + (1.0 - p.z) * 0.05 + sin(flowTime * 0.60) * 0.042;
        float twistCos = cos(twist);
        float twistSin = sin(twist);
        vec2 q = vec2(
            twistCos * p.x - twistSin * p.y,
            twistSin * p.x + twistCos * p.y
        );

        float center = sin(q.x * 1.64 + flowTime * 0.90) * 0.31;
        center += sin(q.x * 3.1 - p.z * 0.65 - flowTime * 0.58) * 0.060;
        center += (broadNoise - 0.5) * 0.068;
        float localFlow = q.y - center;
        float body = 1.0 - smoothstep(0.18, 0.49, abs(localFlow));
        float core = 1.0 - smoothstep(0.035, 0.25, abs(localFlow));

        float fold = sin(q.x * 3.4 - p.z * 0.8 + flowTime * 0.72) * 0.028;
        float blueLayer = 1.0 - smoothstep(0.025, 0.15, abs(localFlow + 0.115 + fold));
        float violetLayer = 1.0 - smoothstep(0.022, 0.135, abs(localFlow - 0.105 - fold * 0.7));
        float pearlVeil = 1.0 - smoothstep(0.030, 0.18, abs(localFlow + fold * 0.24));
        float echo = (1.0 - smoothstep(0.025, 0.14, abs(localFlow + 0.31))) * 0.22;
        float facing = max(dot(normalize(vNormalView), normalize(vViewDirection)), 0.0);
        float interiorMask = smoothstep(0.035, 0.28, facing);
        float depth = smoothstep(-0.18, 0.92, p.z) * (0.58 + facing * 0.42) * interiorMask;

        float silkPulse = 0.5 + 0.5 * sin(q.x * 3.0 - p.z * 1.3 - flowTime * 1.15);
        vec3 mistBlue = vec3(0.27, 0.58, 0.95);
        vec3 pearlBlue = vec3(0.66, 0.80, 0.98);
        vec3 quietLilac = vec3(0.58, 0.46, 0.92);
        vec3 speakerTint = mix(mistBlue, quietLilac, uSpeaker);
        float colorBlend = smoothstep(-0.26, 0.25, localFlow) * 0.58 + smoothstep(-0.9, 0.9, q.x) * 0.22;
        vec3 liquidColor = mix(mistBlue, quietLilac, clamp(colorBlend, 0.0, 1.0));
        liquidColor = mix(liquidColor, mistBlue, blueLayer * 0.30);
        liquidColor = mix(liquidColor, quietLilac, violetLayer * 0.34);
        liquidColor = mix(liquidColor, pearlBlue, pearlVeil * 0.18);
        liquidColor = mix(liquidColor, speakerTint, uLevel * 0.12);

        float flowOpacity = body * 0.25 + core * 0.15;
        flowOpacity += blueLayer * 0.12 + violetLayer * 0.11 + pearlVeil * 0.025 + echo * 0.11;
        flowOpacity *= depth * (0.92 + silkPulse * 0.16) * (1.0 + uLevel * 0.34);
        vec3 color = mix(vec3(0.91, 0.96, 1.0), liquidColor, clamp(body + echo, 0.0, 1.0));
        gl_FragColor = vec4(color, clamp(flowOpacity, 0.0, 0.62));
    }
`;

type OrbMaterial = ShaderMaterial & {
    uniforms: {
        uTime: { value: number };
        uLevel: { value: number };
        uMotion: { value: number };
        uSpeaker: { value: number };
        uConnected?: { value: number };
    };
};

export class AudioReactiveLiquidOrb {
    private readonly container: HTMLElement;
    private readonly reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    private readonly state: AudioReactiveOrbState = {
        audioLevel: 0,
        speaker: "idle",
        connectionState: "disconnected",
    };
    private renderer: WebGLRenderer | null = null;
    private scene: Scene | null = null;
    private camera: PerspectiveCamera | null = null;
    private group: Group | null = null;
    private shell: Mesh<SphereGeometry, OrbMaterial> | null = null;
    private liquid: Mesh<SphereGeometry, OrbMaterial> | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private animationFrame = 0;
    private startTime = performance.now();
    private lastFrameTime = this.startTime;
    private liquidFlowTime = 0;
    private displayedLevel = 0;
    private disposed = false;

    constructor(container: HTMLElement) {
        this.container = container;
        this.initialize();
        this.reducedMotionQuery.addEventListener("change", this.handleReducedMotionChange);
    }

    setState(next: Partial<AudioReactiveOrbState>): void {
        if (next.audioLevel !== undefined) {
            this.state.audioLevel = clamp(next.audioLevel, 0, 0.82);
        }
        if (next.speaker !== undefined) {
            this.state.speaker = next.speaker;
        }
        if (next.connectionState !== undefined) {
            this.state.connectionState = next.connectionState;
        }
        const visualLevel = Math.pow(this.state.audioLevel, 0.68);
        this.container.style.setProperty("--audio-level", visualLevel.toFixed(3));
        this.container.dataset.speaker = this.state.speaker;
        this.container.dataset.connection = this.state.connectionState;
    }

    dispose(): void {
        this.disposed = true;
        cancelAnimationFrame(this.animationFrame);
        this.resizeObserver?.disconnect();
        this.reducedMotionQuery.removeEventListener("change", this.handleReducedMotionChange);
        this.shell?.geometry.dispose();
        this.shell?.material.dispose();
        this.liquid?.geometry.dispose();
        this.liquid?.material.dispose();
        this.renderer?.dispose();
        this.renderer?.domElement.remove();
        this.shell = null;
        this.liquid = null;
        this.group = null;
        this.renderer = null;
        this.scene = null;
        this.camera = null;
    }

    private initialize(): void {
        try {
            const canvas = document.createElement("canvas");
            const renderer = new WebGLRenderer({
                canvas,
                alpha: true,
                antialias: true,
                powerPreference: "high-performance",
                premultipliedAlpha: true,
            });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.65));
            renderer.outputColorSpace = SRGBColorSpace;
            renderer.setClearColor(0x000000, 0);

            const scene = new Scene();
            const camera = new PerspectiveCamera(33, 1, 0.1, 100);
            camera.position.z = 5.45;
            const group = new Group();

            const liquid = new Mesh(
                new SphereGeometry(1.385, 112, 112),
                this.createMaterial(liquidVertexShader, liquidFragmentShader, false),
            );
            liquid.renderOrder = 0;
            liquid.rotation.set(-0.055, -0.13, 0.025);

            const shell = new Mesh(
                new SphereGeometry(1.42, 128, 128),
                this.createMaterial(shellVertexShader, shellFragmentShader, true),
            );
            shell.renderOrder = 1;
            shell.rotation.set(-0.045, -0.10, 0.018);
            group.add(liquid, shell);
            scene.add(group);

            this.renderer = renderer;
            this.scene = scene;
            this.camera = camera;
            this.group = group;
            this.shell = shell;
            this.liquid = liquid;
            this.container.prepend(canvas);
            this.container.classList.add("orb-webgl-ready");
            this.resizeObserver = new ResizeObserver(this.resize);
            this.resizeObserver.observe(this.container);
            this.resize();
            this.animate();
        } catch (error) {
            console.warn("X-Live liquid orb is using the CSS fallback", error);
            this.container.classList.add("orb-fallback");
        }
    }

    private createMaterial(vertexShader: string, fragmentShader: string, includeConnection: boolean): OrbMaterial {
        const uniforms: OrbMaterial["uniforms"] = {
            uTime: { value: 0 },
            uLevel: { value: 0 },
            uMotion: { value: this.reducedMotionQuery.matches ? 0 : 1 },
            uSpeaker: { value: 0 },
        };
        if (includeConnection) {
            uniforms.uConnected = { value: 0 };
        }
        return new ShaderMaterial({
            vertexShader,
            fragmentShader,
            transparent: true,
            depthWrite: false,
            side: FrontSide,
            blending: NormalBlending,
            uniforms,
        }) as OrbMaterial;
    }

    private resize = (): void => {
        if (!this.renderer || !this.camera) {
            return;
        }
        const width = Math.max(1, this.container.clientWidth);
        const height = Math.max(1, this.container.clientHeight);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height, false);
    };

    private animate = (): void => {
        if (this.disposed || !this.renderer || !this.scene || !this.camera || !this.group || !this.shell || !this.liquid) {
            return;
        }
        const now = performance.now();
        const deltaSeconds = Math.min(0.05, (now - this.lastFrameTime) / 1000);
        this.lastFrameTime = now;
        const targetLevel = this.state.audioLevel;
        const responseSeconds = targetLevel > this.displayedLevel ? 0.16 : 0.68;
        const response = 1 - Math.exp(-deltaSeconds / responseSeconds);
        this.displayedLevel += (targetLevel - this.displayedLevel) * response;
        const visualLevel = Math.pow(clamp(this.displayedLevel, 0, 1), 0.68);

        const elapsed = (now - this.startTime) / 1000;
        const motion = this.reducedMotionQuery.matches ? 0 : 1;
        const speaker = this.state.speaker === "agent" ? 1 : 0;
        const connected = this.state.connectionState === "connected" ? 1 : 0;
        this.liquidFlowTime += deltaSeconds * (0.34 + visualLevel * 0.29) * motion;
        this.updateMaterial(this.shell.material, elapsed, visualLevel, motion, speaker, connected);
        this.updateMaterial(this.liquid.material, this.liquidFlowTime, visualLevel, motion, speaker, connected);

        const idleBreath = motion ? Math.sin(elapsed * 0.48) * 0.0045 : 0;
        const voicedBreath = motion ? Math.sin(elapsed * 1.04) * visualLevel * 0.0065 : 0;
        const scale = 1 + idleBreath + voicedBreath + visualLevel * 0.082;
        this.group.scale.set(
            scale * (1 + visualLevel * 0.014),
            scale * (1 - visualLevel * 0.006),
            scale * (1 + visualLevel * 0.004),
        );
        this.group.position.y = motion ? Math.sin(elapsed * 0.31) * 0.009 + visualLevel * 0.017 : 0;
        if (motion) {
            this.group.rotation.y = Math.sin(elapsed * 0.075) * 0.012;
            this.group.rotation.z = Math.sin(elapsed * 0.105) * 0.008;
            this.liquid.rotation.y = -0.13 + Math.sin(elapsed * (0.13 + visualLevel * 0.04)) * 0.045;
            this.liquid.rotation.z = 0.025 + Math.sin(elapsed * (0.10 + visualLevel * 0.03)) * 0.022;
        }

        this.renderer.render(this.scene, this.camera);
        this.animationFrame = requestAnimationFrame(this.animate);
    };

    private updateMaterial(
        material: OrbMaterial,
        elapsed: number,
        visualLevel: number,
        motion: number,
        speaker: number,
        connected: number,
    ): void {
        material.uniforms.uTime.value = elapsed;
        material.uniforms.uLevel.value = visualLevel;
        material.uniforms.uMotion.value = motion;
        material.uniforms.uSpeaker.value = speaker;
        if (material.uniforms.uConnected) {
            material.uniforms.uConnected.value = connected;
        }
    }

    private handleReducedMotionChange = (): void => {
        this.container.classList.toggle("reduce-motion", this.reducedMotionQuery.matches);
    };
}

function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
}
