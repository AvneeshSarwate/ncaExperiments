/**
 * NCA Live Viewer — WebGPU compute + render for interactive NCA visualization.
 *
 * Ping-pong pattern: nca_step (A→B) then post_alive (B→A).
 * stateA always holds the current state.
 * Render reads stateA and composites RGBA over a background color.
 */

const GRID_W = 72;
const GRID_H = 72;
const CHANNELS = 16;
const VISIBLE_CHANNELS = 4;
const STATE_FLOATS = GRID_W * GRID_H * CHANNELS;
const STATE_BYTES = STATE_FLOATS * 4;
const NCA_SCALE = 8; // pixels per grid cell on the canvas

export const NCA_CANVAS_SIZE = GRID_W * NCA_SCALE; // 576

export class NCAViewer {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;

  private stateA!: GPUBuffer;
  private stateB!: GPUBuffer;
  private readbackBuffer!: GPUBuffer;
  private weightsBuffer!: GPUBuffer;
  private modelBuffers = new Map<string, GPUBuffer>();
  private activeModelName: string | null = null;
  private paramsBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBGL!: GPUBindGroupLayout;

  private ncaPipeline!: GPUComputePipeline;
  private alivePipeline!: GPUComputePipeline;
  private bgForward!: GPUBindGroup;
  private bgReverse!: GPUBindGroup;

  private renderPipeline!: GPURenderPipeline;
  private renderBindGroup!: GPUBindGroup;

  private step = 0;
  private _fireRate = 0.5;
  private _stepsPerFrame = 4;
  private running = false;
  private animId = 0;
  private bgColor: [number, number, number] = [0, 0, 0];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async init(
    initialModelName: string,
    weights: Float32Array,
    computeWgsl: string,
    renderWgsl: string,
  ): Promise<void> {
    // --- device + canvas context ---
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");
    this.device = await adapter.requestDevice();

    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format, alphaMode: "opaque" });

    // --- buffers ---
    const makeStorage = () =>
      this.device.createBuffer({
        size: STATE_BYTES,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    this.stateA = makeStorage();
    this.stateB = makeStorage();
    this.readbackBuffer = this.device.createBuffer({
      size: STATE_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Render uniforms: vec4f bg (16 bytes) + vec2u dims (8 bytes) + 8 pad = 32 bytes
    this.renderUniformBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uploadRenderUniforms();

    // --- compute pipelines ---
    const computeModule = this.device.createShaderModule({ code: computeWgsl });
    this.computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeBGL] });

    this.ncaPipeline = this.device.createComputePipeline({
      layout: computeLayout,
      compute: { module: computeModule, entryPoint: "nca_step" },
    });
    this.alivePipeline = this.device.createComputePipeline({
      layout: computeLayout,
      compute: { module: computeModule, entryPoint: "post_alive" },
    });

    // --- render pipeline ---
    const renderModule = this.device.createShaderModule({ code: renderWgsl });
    const renderBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: { module: renderModule, entryPoint: "vs" },
      fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.stateA } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer } },
      ],
    });

    this.addModel(initialModelName, weights);
    await this.setModel(initialModelName);

    // --- init seed ---
    this.reset();
  }

  addModel(name: string, weights: Float32Array): void {
    const existing = this.modelBuffers.get(name);
    if (existing) existing.destroy();

    const buffer = this.bufferWithData(weights, GPUBufferUsage.STORAGE);
    this.modelBuffers.set(name, buffer);

    if (this.activeModelName === name) {
      this.weightsBuffer = buffer;
      this.rebuildComputeBindGroups();
    }
  }

  async setModel(name: string, preserveHiddenState = true): Promise<void> {
    const nextBuffer = this.modelBuffers.get(name);
    if (!nextBuffer) {
      throw new Error(`Unknown model: ${name}`);
    }

    const wasRunning = this.running;
    if (wasRunning) this.stop();

    let nextState: Float32Array | null = null;
    if (!preserveHiddenState) {
      nextState = await this.readState();
      this.zeroHiddenChannels(nextState);
    }

    this.activeModelName = name;
    this.weightsBuffer = nextBuffer;
    this.rebuildComputeBindGroups();

    if (nextState) {
      const bytes = new Uint8Array(nextState.byteLength);
      bytes.set(new Uint8Array(nextState.buffer, nextState.byteOffset, nextState.byteLength));
      this.device.queue.writeBuffer(this.stateA, 0, bytes);
    }

    if (wasRunning) this.start();
  }

  get modelName(): string | null {
    return this.activeModelName;
  }

  reset(): void {
    const data = new Float32Array(STATE_FLOATS);
    const cx = Math.floor(GRID_W / 2);
    const cy = Math.floor(GRID_H / 2);
    const base = (cy * GRID_W + cx) * CHANNELS;
    for (let c = 3; c < CHANNELS; c++) data[base + c] = 1.0;
    this.device.queue.writeBuffer(this.stateA, 0, data);
    this.step = 0;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.animId = requestAnimationFrame(() => this.frame());
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.animId);
  }

  /** Zero cells within a circle. cx/cy are in canvas pixel coords. */
  damage(canvasX: number, canvasY: number, radius: number): void {
    const gx = Math.floor(canvasX / NCA_SCALE);
    const gy = Math.floor(canvasY / NCA_SCALE);
    const gr = Math.ceil(radius / NCA_SCALE);
    const gr2 = (radius / NCA_SCALE) * (radius / NCA_SCALE);
    const zeros = new Float32Array(CHANNELS);

    for (let dy = -gr; dy <= gr; dy++) {
      for (let dx = -gr; dx <= gr; dx++) {
        if (dx * dx + dy * dy > gr2) continue;
        const x = gx + dx;
        const y = gy + dy;
        if (x < 0 || x >= GRID_W || y < 0 || y >= GRID_H) continue;
        const offset = ((y * GRID_W + x) * CHANNELS) * 4;
        this.device.queue.writeBuffer(this.stateA, offset, zeros);
      }
    }
  }

  set stepsPerFrame(n: number) { this._stepsPerFrame = n; }
  set fireRate(r: number) { this._fireRate = r; }

  setBgColor(r: number, g: number, b: number): void {
    this.bgColor = [r, g, b];
    if (this.renderUniformBuffer) this.uploadRenderUniforms();
  }

  destroy(): void {
    this.stop();
    this.stateA.destroy();
    this.stateB.destroy();
    this.readbackBuffer.destroy();
    for (const buffer of this.modelBuffers.values()) {
      buffer.destroy();
    }
    this.modelBuffers.clear();
    this.paramsBuffer.destroy();
    this.renderUniformBuffer.destroy();
  }

  // --- private ---

  private frame(): void {
    const wgX = Math.ceil(GRID_W / 8);
    const wgY = Math.ceil(GRID_H / 8);

    // Run N NCA steps (separate submits so params.step varies per step)
    for (let i = 0; i < this._stepsPerFrame; i++) {
      this.uploadParams();
      const enc = this.device.createCommandEncoder();
      const p1 = enc.beginComputePass();
      p1.setPipeline(this.ncaPipeline);
      p1.setBindGroup(0, this.bgForward);
      p1.dispatchWorkgroups(wgX, wgY);
      p1.end();
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.alivePipeline);
      p2.setBindGroup(0, this.bgReverse);
      p2.dispatchWorkgroups(wgX, wgY);
      p2.end();
      this.device.queue.submit([enc.finish()]);
      this.step++;
    }

    // Render
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: this.bgColor[0], g: this.bgColor[1], b: this.bgColor[2], a: 1 },
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
        },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);

    if (this.running) {
      this.animId = requestAnimationFrame(() => this.frame());
    }
  }

  private uploadParams(): void {
    const buf = new ArrayBuffer(16);
    new Uint32Array(buf, 0, 1)[0] = this.step;
    new Float32Array(buf, 4, 1)[0] = this._fireRate;
    new Uint32Array(buf, 8, 1)[0] = GRID_W;
    new Uint32Array(buf, 12, 1)[0] = GRID_H;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buf);
  }

  private uploadRenderUniforms(): void {
    const buf = new ArrayBuffer(32);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = this.bgColor[0];
    f32[1] = this.bgColor[1];
    f32[2] = this.bgColor[2];
    f32[3] = 0;
    u32[4] = GRID_W;
    u32[5] = GRID_H;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, buf);
  }

  private rebuildComputeBindGroups(): void {
    const bindEntries = (read: GPUBuffer, write: GPUBuffer): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: read } },
      { binding: 1, resource: { buffer: write } },
      { binding: 2, resource: { buffer: this.weightsBuffer } },
      { binding: 3, resource: { buffer: this.paramsBuffer } },
    ];

    this.bgForward = this.device.createBindGroup({
      layout: this.computeBGL,
      entries: bindEntries(this.stateA, this.stateB),
    });
    this.bgReverse = this.device.createBindGroup({
      layout: this.computeBGL,
      entries: bindEntries(this.stateB, this.stateA),
    });
  }

  private async readState(): Promise<Float32Array> {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.stateA, 0, this.readbackBuffer, 0, STATE_BYTES);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
    this.readbackBuffer.unmap();
    return copy;
  }

  private zeroHiddenChannels(state: Float32Array): void {
    for (let base = 0; base < state.length; base += CHANNELS) {
      state.fill(0, base + VISIBLE_CHANNELS, base + CHANNELS);
    }
  }

  private bufferWithData(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(data.byteLength, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }
}
