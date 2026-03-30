#include <metal_stdlib>
using namespace metal;

// NCA-specific Metal compute kernels for 72x72 grid, 16 channels, batch 8.
// All tensors in NCHW layout: index = ((b*C + c)*H + y)*W + x

constant int B = 8;
constant int C = 16;
constant int H = 72;
constant int W = 72;
constant int HW = H * W;      // 5184
constant int PERC = 48;
constant int HIDDEN = 128;
constant uint REDUCE_THREADS = 256;

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

// Perception fused per input channel: writes identity, sobel_x, and sobel_y together.
kernel void perceive_x3(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*16+inC)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;  // b * C + inC
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int inC = bc % C;
    int spatialIdx = ((b * C + inC) * H + y) * W + x;

    float identity = input[spatialIdx];
    float sx = 0.0f;
    float sy = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int syIdx = y + ky;
            int sxIdx = x + kx;
            float val = 0.0f;
            if (syIdx >= 0 && syIdx < H && sxIdx >= 0 && sxIdx < W) {
                val = input[((b * C + inC) * H + syIdx) * W + sxIdx];
            }
            int k = (ky + 1) * 3 + (kx + 1);
            sx += val * sobel_x[k];
            sy += val * sobel_y[k];
        }
    }

    int outBase = ((b * PERC + inC * 3) * H + y) * W + x;
    output[outBase + 0 * HW] = identity;
    output[outBase + 1 * HW] = sx;
    output[outBase + 2 * HW] = sy;
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

// Element-wise multiply for single-channel [B, 1, H, W] masks.
kernel void mask_multiply(
    device const float* lhs  [[buffer(0)]],
    device const float* rhs  [[buffer(1)]],
    device float* output     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(B * HW)) return;
    output[gid] = lhs[gid] * rhs[gid];
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

// Residual add specialized for fireRate=1 and unit life mask.
kernel void residual_add(
    device const float* state  [[buffer(0)]],
    device const float* delta  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int idx = (bc * H + y) * W + x;
    output[idx] = state[idx] + delta[idx];
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

// Apply a [B,1,H,W] life mask to a [B,C,H,W] tensor.
kernel void apply_life_mask(
    device const float* input    [[buffer(0)]],
    device const float* lifeMask [[buffer(1)]],
    device float* output         [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int idx = (bc * H + y) * W + x;
    int maskIdx = (b * H + y) * W + x;
    output[idx] = input[idx] * lifeMask[maskIdx];
}

// Fused alive-mask computation directly from the alpha channel of a [B,C,H,W] state tensor.
kernel void alive_mask_from_state(
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
            int sy = y + ky;
            int sx = x + kx;
            if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                float val = input[((b * C + 3) * H + sy) * W + sx];
                maxVal = max(maxVal, val);
            }
        }
    }
    output[(b * H + y) * W + x] = maxVal > 0.1f ? 1.0f : 0.0f;
}

// Apply pre/post masks to a [B,C,H,W] tensor in one pass.
kernel void apply_two_masks(
    device const float* input    [[buffer(0)]],
    device const float* preMask  [[buffer(1)]],
    device const float* postMask [[buffer(2)]],
    device float* output         [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int idx = (bc * H + y) * W + x;
    int maskIdx = (b * H + y) * W + x;
    output[idx] = input[idx] * preMask[maskIdx] * postMask[maskIdx];
}

// Apply pre/post masks and also persist the combined life mask for backward.
kernel void apply_two_masks_store_life(
    device const float* input    [[buffer(0)]],
    device const float* preMask  [[buffer(1)]],
    device const float* postMask [[buffer(2)]],
    device float* output         [[buffer(3)]],
    device float* lifeMask       [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]  // (x, y, b*C+c)
) {
    int x = gid.x;
    int y = gid.y;
    int bc = gid.z;
    if (x >= W || y >= H || bc >= B * C) return;

    int b = bc / C;
    int c = bc % C;
    int idx = (bc * H + y) * W + x;
    int maskIdx = (b * H + y) * W + x;
    float lm = preMask[maskIdx] * postMask[maskIdx];
    if (c == 0) {
        lifeMask[maskIdx] = lm;
    }
    output[idx] = input[idx] * lm;
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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint gid = tgid.x;
    if (gid >= uint(HIDDEN)) return;

    threadgroup float partials[REDUCE_THREADS];
    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int base = (b * HIDDEN + int(gid)) * HW;
        for (uint yx = tid * 4; yx < uint(HW); yx += REDUCE_THREADS * 4) {
            const device float4* row4 = reinterpret_cast<const device float4*>(dFC1Raw + base + int(yx));
            sum += dot(*row4, float4(1.0f));
        }
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dBias[gid] = partials[0];
    }
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

// 1x1 conv forward with transposed weights [Cin, Cout] and 4 output channels per thread.
kernel void conv1x1_forward_x4(
    device const float* input    [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant int& Cin            [[buffer(3)]],
    constant int& Cout           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bblock = gid.z;
    if (x >= W || y >= H) return;

    int coutBlocks = Cout / 4;
    int b_idx = bblock / coutBlocks;
    int block = bblock % coutBlocks;
    if (b_idx >= B) return;

    int coBase = block * 4;
    int spatialIdx = y * W + x;
    float4 sum = float4(0.0f);

    for (int ci = 0; ci < Cin; ci++) {
        float inVal = input[(b_idx * Cin + ci) * HW + spatialIdx];
        const device float4* w4 = reinterpret_cast<const device float4*>(weight_t + ci * Cout + coBase);
        sum += inVal * (*w4);
    }

    output[(b_idx * Cout + coBase + 0) * HW + spatialIdx] = sum.x;
    output[(b_idx * Cout + coBase + 1) * HW + spatialIdx] = sum.y;
    output[(b_idx * Cout + coBase + 2) * HW + spatialIdx] = sum.z;
    output[(b_idx * Cout + coBase + 3) * HW + spatialIdx] = sum.w;
}

// 1x1 conv forward with transposed weights [Cin, Cout] and 8 output channels per thread.
kernel void conv1x1_forward_x8(
    device const float* input    [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant int& Cin            [[buffer(3)]],
    constant int& Cout           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bblock = gid.z;
    if (x >= W || y >= H) return;

    int coutBlocks = Cout / 8;
    int b_idx = bblock / coutBlocks;
    int block = bblock % coutBlocks;
    if (b_idx >= B) return;

    int coBase = block * 8;
    int spatialIdx = y * W + x;
    float4 sum0 = float4(0.0f);
    float4 sum1 = float4(0.0f);

    for (int ci = 0; ci < Cin; ci++) {
        float inVal = input[(b_idx * Cin + ci) * HW + spatialIdx];
        const device float4* w0 = reinterpret_cast<const device float4*>(weight_t + ci * Cout + coBase);
        const device float4* w1 = reinterpret_cast<const device float4*>(weight_t + ci * Cout + coBase + 4);
        sum0 += inVal * (*w0);
        sum1 += inVal * (*w1);
    }

    output[(b_idx * Cout + coBase + 0) * HW + spatialIdx] = sum0.x;
    output[(b_idx * Cout + coBase + 1) * HW + spatialIdx] = sum0.y;
    output[(b_idx * Cout + coBase + 2) * HW + spatialIdx] = sum0.z;
    output[(b_idx * Cout + coBase + 3) * HW + spatialIdx] = sum0.w;
    output[(b_idx * Cout + coBase + 4) * HW + spatialIdx] = sum1.x;
    output[(b_idx * Cout + coBase + 5) * HW + spatialIdx] = sum1.y;
    output[(b_idx * Cout + coBase + 6) * HW + spatialIdx] = sum1.z;
    output[(b_idx * Cout + coBase + 7) * HW + spatialIdx] = sum1.w;
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

// 1x1 conv data gradient with transposed weights [Cin, Cout] and float4 accumulation over Cout.
kernel void conv1x1_data_grad_t4(
    device const float* dOutput  [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* dInput         [[buffer(2)]],
    constant int& Cin            [[buffer(3)]],
    constant int& Cout           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bci = gid.z;
    if (x >= W || y >= H) return;
    int b_idx = bci / Cin;
    int ci = bci % Cin;
    if (b_idx >= B) return;

    int spatialIdx = y * W + x;
    float sum = 0.0f;
    const device float* wRow = weight_t + ci * Cout;

    for (int co = 0; co < Cout; co += 4) {
        float4 dout4 = float4(
            dOutput[(b_idx * Cout + co + 0) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 1) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 2) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 3) * HW + spatialIdx]
        );
        const device float4* w4 = reinterpret_cast<const device float4*>(wRow + co);
        sum += dot(dout4, *w4);
    }

    dInput[(b_idx * Cin + ci) * HW + spatialIdx] = sum;
}

// 1x1 conv data gradient fused with ReLU backward.
kernel void conv1x1_data_grad_relu_t4(
    device const float* dOutput  [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device const float* hidden   [[buffer(2)]],
    device float* dInput         [[buffer(3)]],
    constant int& Cin            [[buffer(4)]],
    constant int& Cout           [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    int bci = gid.z;
    if (x >= W || y >= H) return;
    int b_idx = bci / Cin;
    int ci = bci % Cin;
    if (b_idx >= B) return;

    int spatialIdx = y * W + x;
    int outIdx = (b_idx * Cin + ci) * HW + spatialIdx;
    float sum = 0.0f;
    const device float* wRow = weight_t + ci * Cout;

    for (int co = 0; co < Cout; co += 4) {
        float4 dout4 = float4(
            dOutput[(b_idx * Cout + co + 0) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 1) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 2) * HW + spatialIdx],
            dOutput[(b_idx * Cout + co + 3) * HW + spatialIdx]
        );
        const device float4* w4 = reinterpret_cast<const device float4*>(wRow + co);
        sum += dot(dout4, *w4);
    }

    dInput[outIdx] = hidden[outIdx] > 0.0f ? sum : 0.0f;
}

// Model-specialized tiled data grad for fc1: dPerc from dFC1Raw.
kernel void fc1_data_grad_tiled(
    device const float* dOutput  [[buffer(0)]],  // [B, HIDDEN, HW]
    device const float* weight_t [[buffer(1)]],  // [PERC, HIDDEN]
    device float* dInput         [[buffer(2)]],  // [B, PERC, HW]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int n = int(tgid.x) * 32 + int(tid.x);
    int ci = int(tgid.y) * 8 + int(tid.y);
    int b = int(tgid.z);
    if (b >= B || ci >= PERC || n >= HW) return;

    const device float* wRow = weight_t + ci * HIDDEN;
    float sum = 0.0f;
    for (int co = 0; co < HIDDEN; co += 4) {
        float4 dout4 = float4(
            dOutput[(b * HIDDEN + co + 0) * HW + n],
            dOutput[(b * HIDDEN + co + 1) * HW + n],
            dOutput[(b * HIDDEN + co + 2) * HW + n],
            dOutput[(b * HIDDEN + co + 3) * HW + n]
        );
        const device float4* w4 = reinterpret_cast<const device float4*>(wRow + co);
        sum += dot(dout4, *w4);
    }
    dInput[(b * PERC + ci) * HW + n] = sum;
}

// Model-specialized tiled data grad for fc2 fused with ReLU backward.
kernel void fc2_data_grad_relu_tiled(
    device const float* dOutput  [[buffer(0)]],  // [B, C, HW]
    device const float* weight_t [[buffer(1)]],  // [HIDDEN, C]
    device const float* hidden   [[buffer(2)]],  // [B, HIDDEN, HW]
    device float* dInput         [[buffer(3)]],  // [B, HIDDEN, HW]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int n = int(tgid.x) * 32 + int(tid.x);
    int ci = int(tgid.y) * 8 + int(tid.y);
    int b = int(tgid.z);
    if (b >= B || ci >= HIDDEN || n >= HW) return;

    const device float* wRow = weight_t + ci * C;
    float sum = 0.0f;
    for (int co = 0; co < C; co += 4) {
        float4 dout4 = float4(
            dOutput[(b * C + co + 0) * HW + n],
            dOutput[(b * C + co + 1) * HW + n],
            dOutput[(b * C + co + 2) * HW + n],
            dOutput[(b * C + co + 3) * HW + n]
        );
        const device float4* w4 = reinterpret_cast<const device float4*>(wRow + co);
        sum += dot(dout4, *w4);
    }

    int outIdx = (b * HIDDEN + ci) * HW + n;
    dInput[outIdx] = hidden[outIdx] > 0.0f ? sum : 0.0f;
}

// Model-specialized tiled weight grad for fc1: dFC1W from dFC1Raw and perc.
kernel void fc1_weight_grad_tiled(
    device const float* dOutput [[buffer(0)]],   // [B, HIDDEN, HW]
    device const float* input   [[buffer(1)]],   // [B, PERC, HW]
    device float* dWeight       [[buffer(2)]],   // [HIDDEN, PERC]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int ci = int(tgid.x) * 8 + int(tid.x);
    int co = int(tgid.y) * 8 + int(tid.y);
    if (ci >= PERC || co >= HIDDEN) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int outBase = (b * HIDDEN + co) * HW;
        int inBase = (b * PERC + ci) * HW;
        for (int n = 0; n < HW; n += 4) {
            const device float4* out4 = reinterpret_cast<const device float4*>(dOutput + outBase + n);
            const device float4* in4 = reinterpret_cast<const device float4*>(input + inBase + n);
            sum += dot(*out4, *in4);
        }
    }

    dWeight[co * PERC + ci] = sum;
}

// Model-specialized tiled weight grad for fc2: dFC2W from dResidual and hidden.
kernel void fc2_weight_grad_tiled(
    device const float* dOutput [[buffer(0)]],   // [B, C, HW]
    device const float* input   [[buffer(1)]],   // [B, HIDDEN, HW]
    device float* dWeight       [[buffer(2)]],   // [C, HIDDEN]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    int ci = int(tgid.x) * 8 + int(tid.x);
    int co = int(tgid.y) * 8 + int(tid.y);
    if (ci >= HIDDEN || co >= C) return;

    float sum = 0.0f;
    for (int b = 0; b < B; ++b) {
        int outBase = (b * C + co) * HW;
        int inBase = (b * HIDDEN + ci) * HW;
        for (int n = 0; n < HW; n += 4) {
            const device float4* out4 = reinterpret_cast<const device float4*>(dOutput + outBase + n);
            const device float4* in4 = reinterpret_cast<const device float4*>(input + inBase + n);
            sum += dot(*out4, *in4);
        }
    }

    dWeight[co * HIDDEN + ci] = sum;
}

// 1x1 conv weight gradient: dWeight[co,ci] = sum_b,y,x(dOutput[b,co,y,x] * input[b,ci,y,x])
// Thread: (ci, co) — one thread per weight element
kernel void conv1x1_weight_grad(
    device const float* dOutput [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* dWeight       [[buffer(2)]],
    constant int& Cin           [[buffer(3)]],
    constant int& Cout          [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int ci = int(tgid.x);
    int co = int(tgid.y);
    if (ci >= Cin || co >= Cout) return;

    threadgroup float partials[REDUCE_THREADS];
    float sum = 0.0f;
    for (int b_idx = 0; b_idx < B; ++b_idx) {
        int outBase = (b_idx * Cout + co) * HW;
        int inBase = (b_idx * Cin + ci) * HW;
        for (uint yx = tid * 4; yx < uint(HW); yx += REDUCE_THREADS * 4) {
            const device float4* out4 = reinterpret_cast<const device float4*>(dOutput + outBase + int(yx));
            const device float4* in4 = reinterpret_cast<const device float4*>(input + inBase + int(yx));
            sum += dot(*out4, *in4);
        }
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dWeight[co * Cin + ci] = partials[0];
    }
}
