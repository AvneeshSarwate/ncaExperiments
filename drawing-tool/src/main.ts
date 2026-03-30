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
      <input type="file" id="weights-upload" accept=".bin" multiple hidden>
    </label>
    <label>Model name: <input type="text" id="model-name" value="" placeholder="optional"></label>
    <label>Active model:
      <select id="model-select" disabled>
        <option value="">No models loaded</option>
      </select>
    </label>
    <button id="nca-reset" disabled>Reset</button>
    <label>Steps/frame: <input type="range" id="steps-per-frame" min="1" max="16" value="4" disabled> <span id="spf-label">4</span></label>
    <label>Erase size: <input type="range" id="erase-size" min="8" max="80" value="32" disabled> <span id="erase-label">32</span>px</label>
    <label><input type="checkbox" id="clear-hidden-on-swap" checked> Clear hidden channels on model swap</label>
  </div>
  <div class="info" id="model-status">Switching models preserves RGBA and clears hidden channels.</div>
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
const modelStatusEl = document.getElementById('model-status')!;
const modelNameInput = document.getElementById('model-name') as HTMLInputElement;
const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
const clearHiddenOnSwapInput = document.getElementById('clear-hidden-on-swap') as HTMLInputElement;
let viewer: NCAViewer | null = null;
const loadedModels = new Map<string, Float32Array>();
const EXPECTED_WEIGHTS = 8320;

function makeUniqueModelName(baseName: string, reservedNames: Set<string>): string {
  const trimmed = baseName.trim();
  const stem = trimmed.length > 0 ? trimmed : `model ${reservedNames.size + 1}`;
  if (!reservedNames.has(stem)) return stem;

  let suffix = 2;
  while (reservedNames.has(`${stem} (${suffix})`)) {
    suffix += 1;
  }
  return `${stem} (${suffix})`;
}

function defaultModelName(file: File): string {
  const raw = file.name.replace(/\.[^.]+$/, '');
  return raw && raw !== 'weights' ? raw : 'weights';
}

function refreshModelSelect(activeName?: string): void {
  modelSelect.innerHTML = '';
  for (const name of loadedModels.keys()) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    modelSelect.appendChild(option);
  }

  modelSelect.disabled = loadedModels.size === 0;
  if (activeName && loadedModels.has(activeName)) {
    modelSelect.value = activeName;
  }
}

function updateViewerStatus(): void {
  const swapModeText = clearHiddenOnSwapInput.checked
    ? 'Switching models preserves RGBA and clears hidden channels.'
    : 'Switching models keeps the full current NCA state, including hidden channels.';

  if (!viewer || !viewer.modelName) {
    statusEl.textContent = 'Upload trained weights to start';
    modelStatusEl.textContent = swapModeText;
    return;
  }

  statusEl.textContent = `Running model "${viewer.modelName}" — click the NCA canvas to erase`;
  modelStatusEl.textContent = `${loadedModels.size} model${loadedModels.size === 1 ? '' : 's'} loaded. ${swapModeText}`;
}

document.getElementById('weights-upload')!.addEventListener('change', async (e) => {
  const inputEl = e.target as HTMLInputElement;
  const files = Array.from(inputEl.files ?? []);
  if (files.length === 0) return;

  statusEl.textContent = 'Initializing WebGPU...';

  try {
    const uploadedModels: Array<{ name: string; weights: Float32Array }> = [];
    const reservedNames = new Set(loadedModels.keys());

    for (const file of files) {
      const arrayBuf = await file.arrayBuffer();
      const weights = new Float32Array(arrayBuf);

      if (weights.length !== EXPECTED_WEIGHTS) {
        statusEl.textContent = `Error: ${file.name} expected ${EXPECTED_WEIGHTS} floats, got ${weights.length}`;
        return;
      }

      const requestedName = files.length === 1 && modelNameInput.value.trim().length > 0
        ? modelNameInput.value
        : defaultModelName(file);
      const uniqueName = makeUniqueModelName(requestedName, reservedNames);
      reservedNames.add(uniqueName);
      uploadedModels.push({ name: uniqueName, weights });
    }

    if (!viewer) {
      const first = uploadedModels[0];
      viewer = new NCAViewer(ncaCanvas);
      await viewer.init(first.name, first.weights, computeWgsl, renderWgsl);
      loadedModels.set(first.name, first.weights);

      for (const model of uploadedModels.slice(1)) {
        viewer.addModel(model.name, model.weights);
        loadedModels.set(model.name, model.weights);
      }

      viewer.start();

      (document.getElementById('nca-reset') as HTMLButtonElement).disabled = false;
      (document.getElementById('steps-per-frame') as HTMLInputElement).disabled = false;
      (document.getElementById('erase-size') as HTMLInputElement).disabled = false;
      refreshModelSelect(first.name);
    } else {
      for (const model of uploadedModels) {
        viewer.addModel(model.name, model.weights);
        loadedModels.set(model.name, model.weights);
      }
      refreshModelSelect(viewer.modelName ?? uploadedModels[0].name);
    }

    modelNameInput.value = '';
    updateViewerStatus();
  } catch (err) {
    statusEl.textContent = `Error: ${err}`;
    console.error(err);
  } finally {
    inputEl.value = '';
  }
});

clearHiddenOnSwapInput.addEventListener('change', () => {
  updateViewerStatus();
});

modelSelect.addEventListener('change', async () => {
  if (!viewer || !modelSelect.value) return;

  const nextModel = modelSelect.value;
  statusEl.textContent = `Switching to "${nextModel}"...`;
  modelSelect.disabled = true;

  try {
    await viewer.setModel(nextModel, !clearHiddenOnSwapInput.checked);
    updateViewerStatus();
  } catch (err) {
    statusEl.textContent = `Error: ${err}`;
    console.error(err);
  } finally {
    refreshModelSelect(viewer.modelName ?? nextModel);
  }
});

document.getElementById('nca-reset')!.addEventListener('click', () => {
  viewer?.reset();
  updateViewerStatus();
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
