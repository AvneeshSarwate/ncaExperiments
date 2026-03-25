import './style.css'

const GRID_SIZE = 40;
const SCALE = 12; // each pixel drawn as 12x12 on screen
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
`;

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

// The actual pixel data lives on a tiny offscreen canvas
const buffer = document.createElement('canvas');
buffer.width = GRID_SIZE;
buffer.height = GRID_SIZE;
const bufCtx = buffer.getContext('2d')!;

// Start transparent
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

// Bresenham line to fill gaps when moving fast
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

  // Draw grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= GRID_SIZE; i++) {
    const pos = i * SCALE;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, CANVAS_SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(CANVAS_SIZE, pos);
    ctx.stroke();
  }
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const [x, y] = getGridPos(e);
  drawDot(x, y);
  lastX = x;
  lastY = y;
  render();
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const [x, y] = getGridPos(e);
  if (x !== lastX || y !== lastY) {
    drawLine(lastX, lastY, x, y);
    lastX = x;
    lastY = y;
    render();
  }
});

window.addEventListener('mouseup', () => {
  drawing = false;
  lastX = -1;
  lastY = -1;
});

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

// Initial render (empty grid with lines)
render();
