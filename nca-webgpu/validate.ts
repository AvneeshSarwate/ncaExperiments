/**
 * Deno validation script — runs NCA via WebGPU compute and compares
 * against PyTorch reference outputs.
 *
 * Usage:
 *   deno run --unstable-webgpu --allow-read validate.ts
 *
 * Prerequisites:
 *   1. Train a model:  uv run python train.py --target images/A.png
 *   2. Generate refs:  uv run python nca-webgpu/generate_reference.py
 */

import { NCAEngine } from "./nca-engine.ts";

const WEIGHTS_PATH = "../output/A/weights.bin";
const REF_DIR = "./reference";
const N_STEPS = 10;

async function main() {
  console.log("=== NCA WebGPU Validation ===\n");

  // Load weights
  const weightsBytes = Deno.readFileSync(WEIGHTS_PATH);
  const weights = new Float32Array(
    weightsBytes.buffer,
    weightsBytes.byteOffset,
    weightsBytes.byteLength / 4,
  );
  console.log(`Loaded weights: ${weights.length} floats (${weightsBytes.byteLength} bytes)`);

  // Load WGSL shader
  const wgsl = new TextDecoder().decode(Deno.readFileSync("./nca.wgsl"));

  // Init engine (fire_rate=1.0 for deterministic comparison)
  const engine = new NCAEngine({ fireRate: 1.0 });
  await engine.init(wgsl, weights);
  console.log("Engine initialised (fire_rate=1.0, deterministic mode)\n");

  // Load seed reference and verify
  const seedRef = loadRef(0);
  const seedGpu = await engine.readState();
  const seedErr = compareStates(seedGpu, seedRef, "Seed (step 0)");

  if (seedErr > 1e-6) {
    console.error("Seed mismatch — aborting.");
    engine.destroy();
    Deno.exit(1);
  }

  // Run steps and compare
  let maxErrAll = 0;
  for (let i = 1; i <= N_STEPS; i++) {
    engine.runStep();

    const gpuState = await engine.readState();
    const refState = loadRef(i);
    const err = compareStates(gpuState, refState, `Step ${i}`);
    maxErrAll = Math.max(maxErrAll, err);
  }

  console.log(`\n=== Summary ===`);
  console.log(`Max absolute error across all ${N_STEPS} steps: ${maxErrAll.toExponential(4)}`);
  if (maxErrAll < 1e-3) {
    console.log("PASS — WebGPU output matches PyTorch within tolerance.");
  } else if (maxErrAll < 1e-1) {
    console.log("MARGINAL — differences above 1e-3, check numerical precision.");
  } else {
    console.log("FAIL — large discrepancy, shader likely has a bug.");
  }

  engine.destroy();
}

function loadRef(step: number): Float32Array {
  const path = `${REF_DIR}/step_${String(step).padStart(3, "0")}.bin`;
  const bytes = Deno.readFileSync(path);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function compareStates(
  gpu: Float32Array,
  ref: Float32Array,
  label: string,
): number {
  if (gpu.length !== ref.length) {
    console.error(`${label}: LENGTH MISMATCH gpu=${gpu.length} ref=${ref.length}`);
    return Infinity;
  }

  let maxErr = 0;
  let sumErr = 0;
  let maxErrIdx = 0;
  const C = 16;
  const W = 72;

  for (let i = 0; i < gpu.length; i++) {
    const err = Math.abs(gpu[i] - ref[i]);
    sumErr += err;
    if (err > maxErr) {
      maxErr = err;
      maxErrIdx = i;
    }
  }

  const avgErr = sumErr / gpu.length;

  // Decode the max-error location
  const c = maxErrIdx % C;
  const pixel = Math.floor(maxErrIdx / C);
  const x = pixel % W;
  const y = Math.floor(pixel / W);

  console.log(
    `  ${label.padEnd(12)} max_err=${maxErr.toExponential(3)}  ` +
    `avg_err=${avgErr.toExponential(3)}  ` +
    `worst@(${x},${y},ch${c}) gpu=${gpu[maxErrIdx].toFixed(6)} ref=${ref[maxErrIdx].toFixed(6)}`,
  );

  return maxErr;
}

main();
