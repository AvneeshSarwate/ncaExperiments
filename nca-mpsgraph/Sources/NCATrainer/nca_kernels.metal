#include <metal_stdlib>
using namespace metal;

// NCA-specific Metal compute kernels for 72x72 grid, 16 channels, batch 8.
// All tensors in NCHW layout: index = ((b*C + c)*H + y)*W + x

constant int B = 8;
constant int C = 16;
constant int H = 72;
constant int W = 72;
constant int HW = H * W;      // 5184
constant int CHW = C * HW;    // 82944
constant int PERC = 48;
constant int HIDDEN = 128;

// Sobel/identity perception kernels (hardcoded, divided by 8)
constant float sobel_x[9] = {-0.125f, 0.0f, 0.125f, -0.25f, 0.0f, 0.25f, -0.125f, 0.0f, 0.125f};
constant float sobel_y[9] = {-0.125f, -0.25f, -0.125f, 0.0f, 0.0f, 0.0f, 0.125f, 0.25f, 0.125f};

// ---- FORWARD KERNELS ----

// Perception: depthwise 3x3 conv with identity + sobel_x + sobel_y
// Input: [B, 16, 72, 72] → Output: [B, 48, 72, 72]
// Thread layout: one thread per (b, output_channel, y, x)
kernel void perceive(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*48+outC)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;  // b * PERC + outC
    if (x >= W || y >= H || bc >= B * PERC) return;

    int b = bc / PERC;
    int outC = bc % PERC;
    int inC = outC / 3;     // which input channel
    int filterIdx = outC % 3; // 0=identity, 1=sobel_x, 2=sobel_y

    float result = 0.0f;

    if (filterIdx == 0) {
        // Identity: just the center value
        result = input[((b * C + inC) * H + y) * W + x];
    } else {
        // Sobel convolution (cross-correlation)
        constant float* kern = (filterIdx == 1) ? sobel_x : sobel_y;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int sy = y + ky;
                int sx = x + kx;
                float val = 0.0f;
                if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                    val = input[((b * C + inC) * H + sy) * W + sx];
                }
                result += val * kern[(ky + 1) * 3 + (kx + 1)];
            }
        }
    }

    output[((b * PERC + outC) * H + y) * W + x] = result;
}

// Bias + ReLU: output = max(0, input + bias)
// Input: [B, 128, 72, 72], Bias: [128] → Output: [B, 128, 72, 72]
kernel void bias_relu(
    device const float* input  [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    device float* output       [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*128+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * HIDDEN) return;

    int c = bc % HIDDEN;
    int idx = (bc * H + y) * W + x;
    float val = input[idx] + bias[c];
    output[idx] = val > 0.0f ? val : 0.0f;
}

// Max pool 3x3, stride 1, same padding, single channel
// Input: [B, 1, 72, 72] → Output: [B, 1, 72, 72]
kernel void max_pool_3x3(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b)
) {
    int x = gid.x;
    int y = gid.y;
    int b = gid.z;
    if (x >= W || y >= H || b >= B) return;

    float maxVal = -1e30f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int sy = y + ky, sx = x + kx;
            if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                float val = input[(b * H + sy) * W + sx];
                maxVal = max(maxVal, val);
            }
        }
    }
    output[(b * H + y) * W + x] = maxVal;
}

// Alive mask: output = (input > 0.1) ? 1.0 : 0.0
// Input/Output: [B, 1, 72, 72]
kernel void threshold_mask(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(B * HW)) return;
    output[gid] = input[gid] > 0.1f ? 1.0f : 0.0f;
}

// Masked residual + life mask: output = (state + delta * fireMask) * lifeMask
// state: [B,16,H,W], delta: [B,16,H,W], fireMask: [B,1,H,W], lifeMask: [B,1,H,W]
// fireMask and lifeMask broadcast over channels
kernel void masked_residual(
    device const float* state    [[buffer(0)]],
    device const float* delta    [[buffer(1)]],
    device const float* fireMask [[buffer(2)]],
    device const float* lifeMask [[buffer(3)]],
    device float* output         [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int idx = (bc * H + y) * W + x;
    int maskIdx = (b * H + y) * W + x;  // [B, 1, H, W] index

    float fm = fireMask[maskIdx];
    float lm = lifeMask[maskIdx];
    output[idx] = (state[idx] + delta[idx] * fm) * lm;
}

// Slice alpha channel: output[b,0,y,x] = input[b,3,y,x]
kernel void slice_alpha(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b)
) {
    int x = gid.x;
    int y = gid.y;
    int b = gid.z;
    if (x >= W || y >= H || b >= B) return;
    output[(b * H + y) * W + x] = input[((b * C + 3) * H + y) * W + x];
}

// ---- BACKWARD KERNELS ----

// Backward of masked_residual: given dOutput, compute dUpdated = dOutput * lifeMask
// and dDelta = dOutput * lifeMask * fireMask
// Also computes dState_direct = dOutput * lifeMask (same as dUpdated)
kernel void backward_masked_residual(
    device const float* dOutput  [[buffer(0)]],
    device const float* fireMask [[buffer(1)]],
    device const float* lifeMask [[buffer(2)]],
    device float* dUpdated       [[buffer(3)]],  // = dOutput * lifeMask [B,C,H,W]
    device float* dDelta         [[buffer(4)]],  // = dOutput * lifeMask * fireMask [B,C,H,W]
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int idx = (bc * H + y) * W + x;
    int maskIdx = (b * H + y) * W + x;

    float du = dOutput[idx] * lifeMask[maskIdx];
    dUpdated[idx] = du;
    dDelta[idx] = du * fireMask[maskIdx];
}

// ReLU backward: dInput = dOutput * (hidden > 0)
kernel void relu_backward(
    device const float* dOutput [[buffer(0)]],
    device const float* hidden  [[buffer(1)]],
    device float* dInput        [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(B * HIDDEN * HW)) return;
    dInput[gid] = hidden[gid] > 0.0f ? dOutput[gid] : 0.0f;
}

// Bias gradient: sum dFC1Raw over batch, H, W → [128]
// Input: [B, 128, H, W] → Output: [128]
kernel void bias_grad(
    device const float* dFC1Raw [[buffer(0)]],
    device float* dBias         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]  // channel index
) {
    if (gid >= uint(HIDDEN)) return;
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                sum += dFC1Raw[((b * HIDDEN + int(gid)) * H + y) * W + x];
            }
        }
    }
    dBias[gid] = sum;
}

// Backward of perceive: transpose depthwise conv with fixed kernels
// Input: dPerc [B, 48, H, W] → Output: dState [B, 16, H, W]
// For each input channel c, sum contributions from the 3 output channels (identity, sobel_x, sobel_y)
kernel void perceive_backward(
    device const float* dPerc  [[buffer(0)]],
    device float* dState       [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int c = bc % C;

    float result = 0.0f;

    // Identity filter contribution: dPerc[b, c*3+0, y, x]
    result += dPerc[((b * PERC + c * 3) * H + y) * W + x];

    // Sobel_x filter: transpose convolution = flip kernel and convolve
    // Flipped sobel_x[ky][kx] = sobel_x[2-ky][2-kx]
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int sy = y + ky, sx = x + kx;
            if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                // Transposed kernel: flip both axes
                float kval_x = sobel_x[(1 - ky) * 3 + (1 - kx)];
                float kval_y = sobel_y[(1 - ky) * 3 + (1 - kx)];
                float dp_x = dPerc[((b * PERC + c * 3 + 1) * H + sy) * W + sx];
                float dp_y = dPerc[((b * PERC + c * 3 + 2) * H + sy) * W + sx];
                result += dp_x * kval_x + dp_y * kval_y;
            }
        }
    }

    dState[(bc * H + y) * W + x] = result;
}

// Element-wise addition: C = A + B
kernel void elementwise_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    C[gid] = A[gid] + B[gid];
}

// ---- 1x1 CONVOLUTION KERNELS (NCHW layout, no reshape needed) ----

// 1x1 conv forward: output[b,co,y,x] = sum_ci(input[b,ci,y,x] * weight[co,ci])
// Input: [B, Cin, H, W], Weight: [Cout, Cin] (OIHW with 1x1), Output: [B, Cout, H, W]
// Thread: (x, y, b*Cout + co)
kernel void conv1x1_forward(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant int& Cin           [[buffer(3)]],
    constant int& Cout          [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bco = gid.z;
    if (x >= W || y >= H) return;
    int totalOut = bco; // check bounds in dispatch
    int b_idx = totalOut / Cout;
    int co = totalOut % Cout;
    if (b_idx >= B) return;

    float sum = 0.0f;
    int spatialIdx = y * W + x;
    for (int ci = 0; ci < Cin; ci++) {
        sum += input[(b_idx * Cin + ci) * HW + spatialIdx] * weight[co * Cin + ci];
    }
    output[(b_idx * Cout + co) * HW + spatialIdx] = sum;
}

// 1x1 conv data gradient: dInput[b,ci,y,x] = sum_co(dOutput[b,co,y,x] * weight[co,ci])
kernel void conv1x1_data_grad(
    device const float* dOutput [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* dInput        [[buffer(2)]],
    constant int& Cin           [[buffer(3)]],
    constant int& Cout          [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bci = gid.z;
    if (x >= W || y >= H) return;
    int b_idx = bci / Cin;
    int ci = bci % Cin;
    if (b_idx >= B) return;

    float sum = 0.0f;
    int spatialIdx = y * W + x;
    for (int co = 0; co < Cout; co++) {
        sum += dOutput[(b_idx * Cout + co) * HW + spatialIdx] * weight[co * Cin + ci];
    }
    dInput[(b_idx * Cin + ci) * HW + spatialIdx] = sum;
}

// 1x1 conv weight gradient: dWeight[co,ci] = sum_b,y,x(dOutput[b,co,y,x] * input[b,ci,y,x])
// Thread: (ci, co) — one thread per weight element
kernel void conv1x1_weight_grad(
    device const float* dOutput [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* dWeight       [[buffer(2)]],
    constant int& Cin           [[buffer(3)]],
    constant int& Cout          [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]  // (ci, co)
) {
    int ci = gid.x;
    int co = gid.y;
    if (ci >= Cin || co >= Cout) return;

    float sum = 0.0f;
    for (int b_idx = 0; b_idx < B; b_idx++) {
        for (int yx = 0; yx < HW; yx++) {
            sum += dOutput[(b_idx * Cout + co) * HW + yx] * input[(b_idx * Cin + ci) * HW + yx];
        }
    }
    dWeight[co * Cin + ci] = sum;
}
