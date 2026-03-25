import './style.css'
import { NCAViewer, NCA_CANVAS_SIZE } from './nca-viewer'
import computeWgsl from './nca-compute.wgsl?raw'
import renderWgsl from './nca-render.wgsl?raw'

// ---- Drawing Tool ----

const GRID_SIZE = 40;
const SCALE = 12;
const CANVAS_SIZE = GRID_SIZE * SCALE;

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <h1>NCA Character Drawing Tool</h1>
  <div class="info">${GRID_SIZE}x${GRID_SIZE} grid — draw characters with thick strokes (4-6px)</div>
  <div class="canvas-container">
    <canvas id="canvas" width="${CANVAS_SIZE}" height="${CANVAS_SIZE}"></canvas>
  </div>
  <div class="controls">
    <label>Color: <input type="color" id="color" value="#ffffff"></label>
    <label>Size: <input type="range" id="brush-size" min="1" max="8" value="5"> <span id="brush-size-label">5</span>px</label>
    <label>Filename: <input type="text" id="filename" value="A"></label>
    <button id="download">Download PNG</button>
    <button id="clear">Clear</button>
  </div>

  <div class="divider"></div>

  <h2>NCA Live Viewer</h2>
  <div class="info" id="nca-status">Upload trained weights to start</div>
  <div class="nca-controls">
    <label class="file-upload-btn">
      Load weights.bin
      <input type="file" id="weights-upload" accept=".bin" hidden>
    </label>
    <button id="nca-reset" disabled>Reset</button>
    <label>Steps/frame: <input type="range" id="steps-per-frame" min="1" max="16" value="4" disabled> <span id="spf-label">4</span></label>
    <label>Erase size: <input type="range" id="erase-size" min="8" max="80" value="32" disabled> <span id="erase-label">32</span>px</label>
  </div>
  <div class="nca-canvas-container">
    <canvas id="nca-canvas" width="${NCA_CANVAS_SIZE}" height="${NCA_CANVAS_SIZE}"></canvas>
  </div>
  <div class="info">Click and drag on the NCA canvas to erase — it repairs itself</div>
`;

// ---- Drawing canvas setup ----

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const buffer = document.createElement('canvas');
buffer.width = GRID_SIZE;
buffer.height = GRID_SIZE;
const bufCtx = buffer.getContext('2d')!;
bufCtx.clearRect(0, 0, GRID_SIZE, GRID_SIZE);

let drawing = false;
let lastX = -1;
let lastY = -1;

function getGridPos(e: MouseEvent): [number, number] {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / SCALE);
  const y = Math.floor((e.clientY - rect.top) / SCALE);
  return [
    Math.max(0, Math.min(GRID_SIZE - 1, x)),
    Math.max(0, Math.min(GRID_SIZE - 1, y)),
  ];
}

function getColor(): string {
  return (document.getElementById('color') as HTMLInputElement).value;
}

function getBrushSize(): number {
  return parseInt((document.getElementById('brush-size') as HTMLInputElement).value);
}

function drawDot(cx: number, cy: number) {
  const size = getBrushSize();
  const radius = size / 2;
  bufCtx.fillStyle = getColor();
  for (let dy = -Math.ceil(radius); dy <= Math.ceil(radius); dy++) {
    for (let dx = -Math.ceil(radius); dx <= Math.ceil(radius); dx++) {
      if (dx * dx + dy * dy <= radius * radius) {
        const px = cx + dx;
        const py = cy + dy;
        if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
          bufCtx.fillRect(px, py, 1, 1);
        }
      }
    }
  }
}

function drawLine(x0: number, y0: number, x1: number, y1: number) {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;

  while (true) {
    drawDot(x0, y0);
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x0 += sx; }
    if (e2 < dx) { err += dx; y0 += sy; }
  }
}

function render() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(buffer, 0, 0, CANVAS_SIZE, CANVAS_SIZE);

  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= GRID_SIZE; i++) {
    const pos = i * SCALE;
    ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, CANVAS_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(CANVAS_SIZE, pos); ctx.stroke();
  }
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const [x, y] = getGridPos(e);
  drawDot(x, y);
  lastX = x; lastY = y;
  render();
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const [x, y] = getGridPos(e);
  if (x !== lastX || y !== lastY) {
    drawLine(lastX, lastY, x, y);
    lastX = x; lastY = y;
    render();
  }
});

window.addEventListener('mouseup', () => { drawing = false; lastX = -1; lastY = -1; });

document.getElementById('clear')!.addEventListener('click', () => {
  bufCtx.clearRect(0, 0, GRID_SIZE, GRID_SIZE);
  render();
});

document.getElementById('download')!.addEventListener('click', () => {
  const filename = (document.getElementById('filename') as HTMLInputElement).value || 'char';
  const link = document.createElement('a');
  link.download = `${filename}.png`;
  link.href = buffer.toDataURL('image/png');
  link.click();
});

document.getElementById('brush-size')!.addEventListener('input', (e) => {
  document.getElementById('brush-size-label')!.textContent = (e.target as HTMLInputElement).value;
});

render();

// ---- NCA Viewer ----

const ncaCanvas = document.getElementById('nca-canvas') as HTMLCanvasElement;
const statusEl = document.getElementById('nca-status')!;
let viewer: NCAViewer | null = null;

document.getElementById('weights-upload')!.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  statusEl.textContent = 'Initializing WebGPU...';

  try {
    const arrayBuf = await file.arrayBuffer();
    const weights = new Float32Array(arrayBuf);

    if (weights.length !== 8320) {
      statusEl.textContent = `Error: expected 8320 floats, got ${weights.length}`;
      return;
    }

    if (viewer) viewer.destroy();

    viewer = new NCAViewer(ncaCanvas);
    await viewer.init(weights, computeWgsl, renderWgsl);
    viewer.start();

    statusEl.textContent = 'Running — click the NCA canvas to erase';

    // Enable controls
    (document.getElementById('nca-reset') as HTMLButtonElement).disabled = false;
    (document.getElementById('steps-per-frame') as HTMLInputElement).disabled = false;
    (document.getElementById('erase-size') as HTMLInputElement).disabled = false;
  } catch (err) {
    statusEl.textContent = `Error: ${err}`;
    console.error(err);
  }
});

document.getElementById('nca-reset')!.addEventListener('click', () => {
  viewer?.reset();
});

document.getElementById('steps-per-frame')!.addEventListener('input', (e) => {
  const val = parseInt((e.target as HTMLInputElement).value);
  document.getElementById('spf-label')!.textContent = String(val);
  if (viewer) viewer.stepsPerFrame = val;
});

document.getElementById('erase-size')!.addEventListener('input', (e) => {
  document.getElementById('erase-label')!.textContent = (e.target as HTMLInputElement).value;
});

// NCA canvas mouse interaction (erase/damage)
let erasing = false;

ncaCanvas.addEventListener('mousedown', (e) => {
  if (!viewer) return;
  erasing = true;
  const rect = ncaCanvas.getBoundingClientRect();
  const scaleX = ncaCanvas.width / rect.width;
  const scaleY = ncaCanvas.height / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top) * scaleY;
  const radius = parseInt((document.getElementById('erase-size') as HTMLInputElement).value) / 2;
  viewer.damage(cx, cy, radius);
});

ncaCanvas.addEventListener('mousemove', (e) => {
  if (!erasing || !viewer) return;
  const rect = ncaCanvas.getBoundingClientRect();
  const scaleX = ncaCanvas.width / rect.width;
  const scaleY = ncaCanvas.height / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top) * scaleY;
  const radius = parseInt((document.getElementById('erase-size') as HTMLInputElement).value) / 2;
  viewer.damage(cx, cy, radius);
});

window.addEventListener('mouseup', () => { erasing = false; });
