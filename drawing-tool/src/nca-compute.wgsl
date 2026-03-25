// NCA Update Compute Shader
// Growing Neural Cellular Automata (Mordvintsev et al., 2020)
//
// Two entry points used in sequence per step:
//   1. nca_step:   perception -> FC1 -> FC2 -> pre-alive mask -> stochastic mask -> residual add
//   2. post_alive: post-alive mask (zeros cells with no alive neighbors after update)
//
// Buffer layout (ping-pong):
//   nca_step:  read stateA (binding 0), write stateB (binding 1)
//   post_alive: read stateB (binding 0), write stateA (binding 1)
//   Result: stateA always holds the current state.

struct Params {
    step: u32,        // current step (used as RNG seed)
    fire_rate: f32,   // stochastic update probability (0.5 in training, 1.0 for deterministic)
    grid_w: u32,
    grid_h: u32,
}

@group(0) @binding(0) var<storage, read>       state_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:   array<f32>;
@group(0) @binding(3) var<uniform>             params:    Params;

const C: u32 = 16u;       // total channels (4 RGBA + 12 hidden)
const HIDDEN: u32 = 128u; // FC1 output / FC2 input width
const PERCEPT: u32 = 48u; // C * 3 (identity + sobel_x + sobel_y per channel)

// Weight buffer offsets (floats, not bytes)
const W1_OFF: u32 = 0u;           // conv1.weight [128, 48, 1, 1] -> 6144 floats
const B1_OFF: u32 = 6144u;        // conv1.bias   [128]           -> 128 floats
const W2_OFF: u32 = 6272u;        // conv2.weight [16, 128, 1, 1] -> 2048 floats
                                   // total: 8320 floats

// --- helpers ---

fn cell_idx(x: u32, y: u32, c: u32) -> u32 {
    return (y * params.grid_w + x) * C + c;
}

fn read_cell(x: i32, y: i32, c: u32) -> f32 {
    // Zero-padding for out-of-bounds (matches PyTorch F.conv2d padding=1)
    if (x < 0 || x >= i32(params.grid_w) || y < 0 || y >= i32(params.grid_h)) {
        return 0.0;
    }
    return state_in[cell_idx(u32(x), u32(y), c)];
}

// PCG hash for per-cell stochastic mask
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand01(x: u32, y: u32, step: u32) -> f32 {
    let seed = x + y * 73u + step * 7919u; // primes to reduce correlation
    return f32(pcg(seed)) / 4294967295.0;
}

// --- nca_step: main NCA update ---

@compute @workgroup_size(8, 8)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.grid_w || y >= params.grid_h) { return; }

    let ix = i32(x);
    let iy = i32(y);
    let base = cell_idx(x, y, 0u);

    // --- Pre-alive mask: max alpha in 3x3 neighborhood > 0.1 ---
    var max_alpha: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            max_alpha = max(max_alpha, read_cell(ix + dx, iy + dy, 3u));
        }
    }

    if (max_alpha <= 0.1) {
        // Dead cell — zero all channels
        for (var c = 0u; c < C; c++) {
            state_out[base + c] = 0.0;
        }
        return;
    }

    // --- Perception: identity + Sobel-X + Sobel-Y per channel ---
    // Matches PyTorch depthwise conv ordering: [ch0_id, ch0_sx, ch0_sy, ch1_id, ...]
    var percept: array<f32, 48>;

    for (var c = 0u; c < C; c++) {
        let p = c * 3u;

        // Identity
        percept[p] = read_cell(ix, iy, c);

        // Sobel-X: outer([1,2,1], [-1,0,1]) / 8
        percept[p + 1u] =
            read_cell(ix - 1, iy - 1, c) * (-0.125) +
            read_cell(ix + 1, iy - 1, c) * ( 0.125) +
            read_cell(ix - 1, iy    , c) * (-0.25 ) +
            read_cell(ix + 1, iy    , c) * ( 0.25 ) +
            read_cell(ix - 1, iy + 1, c) * (-0.125) +
            read_cell(ix + 1, iy + 1, c) * ( 0.125);

        // Sobel-Y: transpose of Sobel-X kernel
        percept[p + 2u] =
            read_cell(ix - 1, iy - 1, c) * (-0.125) +
            read_cell(ix    , iy - 1, c) * (-0.25 ) +
            read_cell(ix + 1, iy - 1, c) * (-0.125) +
            read_cell(ix - 1, iy + 1, c) * ( 0.125) +
            read_cell(ix    , iy + 1, c) * ( 0.25 ) +
            read_cell(ix + 1, iy + 1, c) * ( 0.125);
    }

    // --- FC1: 48 → 128, ReLU ---
    var hidden: array<f32, 128>;
    for (var j = 0u; j < HIDDEN; j++) {
        var sum: f32 = weights[B1_OFF + j]; // bias
        for (var i = 0u; i < PERCEPT; i++) {
            sum += percept[i] * weights[W1_OFF + j * PERCEPT + i];
        }
        hidden[j] = max(sum, 0.0); // ReLU
    }

    // --- FC2: 128 → 16, no bias ---
    var delta: array<f32, 16>;
    for (var c = 0u; c < C; c++) {
        var sum: f32 = 0.0;
        for (var j = 0u; j < HIDDEN; j++) {
            sum += hidden[j] * weights[W2_OFF + c * HIDDEN + j];
        }
        delta[c] = sum;
    }

    // --- Stochastic update mask (same for all channels of this cell) ---
    let fire = rand01(x, y, params.step) <= params.fire_rate;

    // --- Residual add ---
    for (var c = 0u; c < C; c++) {
        var val = state_in[base + c];
        if (fire) {
            val += delta[c];
        }
        state_out[base + c] = val;
    }
}

// --- post_alive: apply post-update alive mask ---
// Reads from the nca_step output and zeros cells with no alive neighbors.

@compute @workgroup_size(8, 8)
fn post_alive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.grid_w || y >= params.grid_h) { return; }

    let ix = i32(x);
    let iy = i32(y);
    let base = cell_idx(x, y, 0u);

    // Check max alpha in 3x3 neighborhood (reading from state_in = nca_step's output)
    var max_alpha: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            max_alpha = max(max_alpha, read_cell(ix + dx, iy + dy, 3u));
        }
    }

    let alive = max_alpha > 0.1;

    for (var c = 0u; c < C; c++) {
        if (alive) {
            state_out[base + c] = state_in[base + c];
        } else {
            state_out[base + c] = 0.0;
        }
    }
}
