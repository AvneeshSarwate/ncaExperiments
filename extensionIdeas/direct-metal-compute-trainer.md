# Direct Metal Compute NCA Trainer

## Goal

Beat PyTorch MPS training speed on Apple Silicon by using direct Metal compute shaders instead of MPSGraph, matching how PyTorch MPS internally dispatches operations.

## Why This Should Be Faster

PyTorch MPS achieves ~5.5 steps/s (181ms/step) on M1 Max for NCA training. It does NOT use MPSGraph for most operations — it uses direct Metal compute shaders dispatched via `MTLComputeCommandEncoder` at ~3μs per dispatch. MPSGraph is only used for convolutions.

Our MPSGraph implementation peaked at 5.2 steps/s (192ms/step) — close but limited by MPSGraph's per-encode overhead (~225μs per call vs PyTorch's ~3μs per dispatch). The entire gap is CPU-side encoding overhead, not GPU compute.

Direct Metal compute would:
- Match PyTorch's ~3μs per-dispatch overhead
- Add per-step op fusion (e.g., perceive kernel hardcodes Sobel values, bias_relu fuses two ops)
- Keep GPU buffers alive across forward/backward (like PyTorch's autograd) — zero-copy intermediate saving
- Use a single persistent `MTLComputeCommandEncoder` with no commit between steps

## Profiling Data (MPSGraph version)

Per training step breakdown at 80 NCA steps, batch 8, 72x72 grid:

### PyTorch MPS (baseline: 181ms/step)
- Forward 80 steps: 62ms (GPU compute, dispatched async)
- Backward 80 steps: 103ms (GPU compute, dispatched async)
- CPU overhead: 16ms (pool sampling, optimizer, writeback)
- GPU work: ~165ms, CPU overhead: ~16ms (fully overlapped with GPU)

### MPSGraph best (192ms/step)
- Forward encode: 64ms (80 runAsync calls, overlapped with GPU)
- Backward encode: 103ms (80 runAsync calls, overlapped with GPU)
- Rest: 12ms (3 gradient readbacks + Adam on CPU)
- GPU work: ~165ms, CPU overhead: ~27ms (partially overlapped)

### Gap analysis
The 11ms gap comes from:
- Forward: 78ms vs 62ms = +16ms (MPSGraph encode overhead per step, 80 × ~200μs)
- Backward: 102ms vs 103ms = -1ms (manual backward matches PyTorch)
- Rest: 12ms vs 16ms = -4ms (fewer readbacks)

## Architecture

### Forward pass (one NCA step)
8 Metal kernel dispatches replacing ~12 PyTorch dispatches:

1. **perceive** — Custom kernel. Depthwise 3x3 conv with hardcoded Sobel/identity values. Input [8,16,72,72] → Output [8,48,72,72]. ~30 lines Metal.
2. **conv1x1_forward** (fc1) — Custom kernel. Per-pixel matmul in NCHW layout. Input [8,48,72,72] × Weight [128,48] → Output [8,128,72,72]. No reshape needed.
3. **bias_relu** — Fused bias add + ReLU. Input [8,128,72,72] + Bias [128] → Output [8,128,72,72].
4. **conv1x1_forward** (fc2) — Same kernel, different sizes. Input [8,128,72,72] × Weight [16,128] → Output [8,16,72,72].
5. **slice_alpha** — Extract channel 3. [8,16,72,72] → [8,1,72,72].
6. **max_pool_3x3** — 3x3 max pool, stride 1, same padding on alpha. [8,1,72,72] → [8,1,72,72].
7. **threshold_mask** — Hard threshold (alpha > 0.1). [8,1,72,72] → [8,1,72,72].
8. **masked_residual** — state + delta * fireMask, then * lifeMask. All element-wise with broadcast.

Steps 5-7 run twice (pre and post alive mask). Fire mask generated from random values (threshold at 0.5).

### Manual backward pass (one NCA step)
~8 Metal kernel dispatches, no forward recomputation:

1. **backward_masked_residual** — dUpdated = dOutput * lifeMask, dDelta = dUpdated * fireMask
2. **conv1x1_data_grad** (fc2) — dHidden from dDelta via transpose weight multiply
3. **conv1x1_weight_grad** (fc2) — dFC2W from dDelta and hidden
4. **relu_backward** — dFC1Raw = dHidden * (hidden > 0)
5. **bias_grad** — dFC1B = sum(dFC1Raw) over batch, H, W → [128]
6. **conv1x1_data_grad** (fc1) — dPerc from dFC1Raw via transpose weight multiply
7. **conv1x1_weight_grad** (fc1) — dFC1W from dFC1Raw and perc
8. **perceive_backward** — dState from dPerc via transposed fixed Sobel convolution
9. **elementwise_add** — dState_total = dUpdated + dState_from_perc

### Saved activations (forward → backward)
Per step, keep GPU buffers alive (zero-copy, like PyTorch autograd):
- state (input): [8,16,72,72] = 2.6MB
- perc: [8,48,72,72] = 7.9MB
- hidden: [8,128,72,72] = 21.2MB
- fireMask: [8,1,72,72] = 0.17MB
- lifeMask: [8,1,72,72] = 0.17MB

Total per step: ~32MB. For 80 steps: ~2.5GB (same as PyTorch).

### Gradient accumulation
Built into backward graph: each backward step takes running accumulator tensors as input and outputs updated accumulators. Only 3 final readbacks (fc1W grad, fc1B grad, fc2W grad = 33KB total).

### Training loop structure
```
for step in 0..<8000:
    // CPU: pool sample, sort, seed, damage (~1ms)

    // GPU: single command buffer for entire step
    encoder = commandBuffer.makeComputeCommandEncoder()

    // Forward: 80 × ~8 dispatches (keep all buffers alive)
    for i in 0..<80:
        dispatch perceive, conv1x1, bias_relu, conv1x1, alive_mask, masked_residual
        save [state, perc, hidden, fireMask, lifeMask] as buffer references

    // Loss gradient: computed on CPU (~1ms, 166K floats)

    // Backward: 80 × ~9 dispatches (read saved buffers, accumulate grads)
    for i in (0..<80).reversed():
        dispatch backward_masked_residual, conv_grads, relu_grad, bias_grad, perceive_backward
        accumulate weight gradients in running buffers

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // CPU: read 3 gradient buffers, normalize, Adam, upload weights (~5ms)
```

Total dispatches per step: ~1360 (80×8 forward + 80×9 backward)
At ~3μs per dispatch: ~4ms CPU encoding
GPU work: ~165ms (same as PyTorch)
Expected total: ~170ms/step = **5.9 steps/s** (7% faster than PyTorch)

## Implementation Status

### Completed
- All Metal compute kernels written and compiled (`nca_kernels.metal`, `nca_kernels.metallib`)
  - perceive, bias_relu, max_pool_3x3, threshold_mask, masked_residual, slice_alpha
  - conv1x1_forward, conv1x1_data_grad, conv1x1_weight_grad
  - backward_masked_residual, relu_backward, bias_grad, perceive_backward, elementwise_add
- Swift dispatch infrastructure (NCAMetal class with pipeline states, buffer management)
- Per-operation PyTorch reference data generated for verification (`reference_ops/`)
- Metal shaders compile successfully, all 14 kernel functions found in metallib

### Blocked
- SPM-built binary crashes (SIGSEGV) during forward pass execution
- Metal device + library + pipeline states initialize correctly
- Crash is after init, during dispatch or buffer access
- Same shaders work fine from `swift -e` (non-SPM context)
- Likely cause: buffer size mismatch or dispatch thread count overflow in the conv1x1_forward kernel
- Debug approach: isolate each kernel in a minimal test, verify one at a time

### Remaining work once crash is fixed
1. Verify forward pass matches PyTorch reference (per-operation comparison)
2. Verify backward pass matches PyTorch reference
3. Wire up the training loop (pool, loss, Adam, weight export)
4. Benchmark against PyTorch MPS
5. Add batch training support (read same YAML config)

## Key Insight: Why MPSGraph Can't Match PyTorch

PyTorch MPS does NOT use MPSGraph as a graph framework. It uses:
- Custom Metal compute shaders (`.metal` files in PyTorch source) for element-wise ops, reductions, random, comparisons
- `MTLComputeCommandEncoder.dispatchThreadgroups()` for each op (~3μs overhead)
- MPS library kernels (MPSMatrixMultiplication, etc.) only for complex ops like convolution
- A persistent command encoder that accumulates without committing
- Autograd saves references to GPU buffers (zero-copy activation caching)

MPSGraph adds ~225μs per encode() call for graph resolution, tensor validation, and buffer management. With 160 encode calls per training step, this adds ~36ms of overhead that PyTorch doesn't have. This is an inherent limitation of the MPSGraph abstraction layer.

The direct Metal approach eliminates this by using the same dispatch pattern as PyTorch — raw `dispatchThreadgroups` calls on a persistent encoder. Combined with fused kernels (perceive, bias_relu), it should be strictly faster than PyTorch.

## Files

- `nca-mpsgraph/Sources/NCATrainer/nca_kernels.metal` — All Metal compute kernels
- `nca-mpsgraph/Sources/NCATrainer/main.swift` — Swift dispatch code (verification mode)
- `nca-mpsgraph/nca_kernels.metallib` — Compiled Metal shader library
- `nca-mpsgraph/reference_ops/` — Per-operation PyTorch reference data for verification
- `nca-mpsgraph/reference/` — Step-by-step NCHW reference outputs
