/**
 * WebGPU NCA Engine — portable across Deno and browser.
 *
 * Runs the Growing Neural Cellular Automata compute shader with a ping-pong
 * buffer pattern:
 *   nca_step:   stateA (read) → stateB (write)
 *   post_alive: stateB (read) → stateA (write)
 * After each step, stateA holds the current state.
 */

// ---- public types ----

export interface NCAEngineOptions {
  gridWidth?: number;
  gridHeight?: number;
  channelCount?: number;
  fireRate?: number;
}

// ---- engine ----

export class NCAEngine {
  readonly gridW: number;
  readonly gridH: number;
  readonly channels: number;
  readonly cellCount: number;
  readonly stateFloats: number;

  private device!: GPUDevice;
  private stateA!: GPUBuffer;
  private stateB!: GPUBuffer;
  private weightsBuffer!: GPUBuffer;
  private paramsBuffer!: GPUBuffer;
  private readbackBuffer!: GPUBuffer;

  private ncaPipeline!: GPUComputePipeline;
  private alivePipeline!: GPUComputePipeline;
  private bgForward!: GPUBindGroup;  // nca_step:  A→B
  private bgReverse!: GPUBindGroup;  // post_alive: B→A

  private step = 0;
  private fireRate: number;

  constructor(opts: NCAEngineOptions = {}) {
    this.gridW = opts.gridWidth ?? 72;
    this.gridH = opts.gridHeight ?? 72;
    this.channels = opts.channelCount ?? 16;
    this.fireRate = opts.fireRate ?? 0.5;
    this.cellCount = this.gridW * this.gridH;
    this.stateFloats = this.cellCount * this.channels;
  }

  /** Initialise GPU resources. Call once before anything else. */
  async init(wgslSource: string, weightsF32: Float32Array): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found");
    this.device = await adapter.requestDevice();

    const stateBytes = this.stateFloats * 4;

    // State ping-pong buffers
    this.stateA = this.device.createBuffer({
      size: stateBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.stateB = this.device.createBuffer({
      size: stateBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Readback buffer (for CPU access)
    this.readbackBuffer = this.device.createBuffer({
      size: stateBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Weights buffer
    this.weightsBuffer = this.createBufferWithData(
      weightsF32,
      GPUBufferUsage.STORAGE,
    );

    // Params uniform (16 bytes: u32 step, f32 fire_rate, u32 grid_w, u32 grid_h)
    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Shader module
    const shaderModule = this.device.createShaderModule({ code: wgslSource });

    // Bind group layout (shared by both pipelines)
    const bgl = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bgl],
    });

    // Two pipelines, two entry points
    this.ncaPipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "nca_step" },
    });
    this.alivePipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "post_alive" },
    });

    // Bind groups with swapped state buffers
    const makeEntries = (
      read: GPUBuffer,
      write: GPUBuffer,
    ): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: read } },
      { binding: 1, resource: { buffer: write } },
      { binding: 2, resource: { buffer: this.weightsBuffer } },
      { binding: 3, resource: { buffer: this.paramsBuffer } },
    ];

    this.bgForward = this.device.createBindGroup({
      layout: bgl,
      entries: makeEntries(this.stateA, this.stateB),
    });
    this.bgReverse = this.device.createBindGroup({
      layout: bgl,
      entries: makeEntries(this.stateB, this.stateA),
    });

    // Initialise stateA with seed
    this.resetToSeed();
  }

  /** Write a seed state: single center pixel with alpha + hidden = 1. */
  resetToSeed(): void {
    const data = new Float32Array(this.stateFloats);
    const cx = Math.floor(this.gridW / 2);
    const cy = Math.floor(this.gridH / 2);
    const base = (cy * this.gridW + cx) * this.channels;
    for (let c = 3; c < this.channels; c++) {
      data[base + c] = 1.0; // alpha=1, hidden channels=1
    }
    this.device.queue.writeBuffer(this.stateA, 0, data);
    this.step = 0;
  }

  /** Write an arbitrary state into the current buffer. */
  setState(data: Float32Array): void {
    this.device.queue.writeBuffer(this.stateA, 0, data);
  }

  /** Run one NCA step (nca_step + post_alive). */
  runStep(): void {
    this.uploadParams();

    const wgX = Math.ceil(this.gridW / 8);
    const wgY = Math.ceil(this.gridH / 8);

    const encoder = this.device.createCommandEncoder();

    // Pass 1: stateA → stateB (NCA update + pre-alive)
    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(this.ncaPipeline);
    pass1.setBindGroup(0, this.bgForward);
    pass1.dispatchWorkgroups(wgX, wgY);
    pass1.end();

    // Pass 2: stateB → stateA (post-alive mask)
    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(this.alivePipeline);
    pass2.setBindGroup(0, this.bgReverse);
    pass2.dispatchWorkgroups(wgX, wgY);
    pass2.end();

    this.device.queue.submit([encoder.finish()]);
    this.step++;
  }

  /** Run N steps without readback (fast batch). */
  runSteps(n: number): void {
    for (let i = 0; i < n; i++) {
      this.runStep();
    }
  }

  /** Read back the current state (stateA) to CPU. Async — waits for GPU. */
  async readState(): Promise<Float32Array> {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.stateA,
      0,
      this.readbackBuffer,
      0,
      this.stateFloats * 4,
    );
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(
      this.readbackBuffer.getMappedRange().slice(0),
    );
    this.readbackBuffer.unmap();
    return copy;
  }

  /** Zero out a rectangular region of the state (all channels). For interactive damage. */
  async damageRect(
    rx: number,
    ry: number,
    rw: number,
    rh: number,
  ): Promise<void> {
    const state = await this.readState();
    for (let y = ry; y < ry + rh && y < this.gridH; y++) {
      for (let x = rx; x < rx + rw && x < this.gridW; x++) {
        const base = (y * this.gridW + x) * this.channels;
        for (let c = 0; c < this.channels; c++) {
          state[base + c] = 0;
        }
      }
    }
    this.device.queue.writeBuffer(this.stateA, 0, state);
  }

  /** Zero a circular region centered at (cx, cy) with given radius. */
  async damageCircle(
    cx: number,
    cy: number,
    radius: number,
  ): Promise<void> {
    const state = await this.readState();
    const r2 = radius * radius;
    const minX = Math.max(0, Math.floor(cx - radius));
    const maxX = Math.min(this.gridW - 1, Math.ceil(cx + radius));
    const minY = Math.max(0, Math.floor(cy - radius));
    const maxY = Math.min(this.gridH - 1, Math.ceil(cy + radius));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy <= r2) {
          const base = (y * this.gridW + x) * this.channels;
          for (let c = 0; c < this.channels; c++) {
            state[base + c] = 0;
          }
        }
      }
    }
    this.device.queue.writeBuffer(this.stateA, 0, state);
  }

  /** Get the underlying GPU device (for advanced usage / render integration). */
  getDevice(): GPUDevice {
    return this.device;
  }

  /** Get the current state buffer (for binding in a render pipeline). */
  getStateBuffer(): GPUBuffer {
    return this.stateA;
  }

  get currentStep(): number {
    return this.step;
  }

  setFireRate(rate: number): void {
    this.fireRate = rate;
  }

  destroy(): void {
    this.stateA.destroy();
    this.stateB.destroy();
    this.weightsBuffer.destroy();
    this.paramsBuffer.destroy();
    this.readbackBuffer.destroy();
  }

  // ---- private ----

  private uploadParams(): void {
    const buf = new ArrayBuffer(16);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = this.step;
    f32[1] = this.fireRate;
    u32[2] = this.gridW;
    u32[3] = this.gridH;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buf);
  }

  private createBufferWithData(
    data: Float32Array,
    usage: GPUBufferUsageFlags,
  ): GPUBuffer {
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
