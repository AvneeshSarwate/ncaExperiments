# Direct Metal Optimization Log

## Scope

This log tracks the optimization attempts made while porting the NCA trainer from the PyTorch reference to direct Metal compute in `nca-mpsgraph/`.

Target:
- match or beat PyTorch MPS on Apple Silicon
- keep the pure-Metal path numerically aligned with the PyTorch reference
- prioritize PyTorch-like per-step dispatch/orchestration first
- defer cross-time-step fusion tricks until per-step parity is reached

## Reference Commands

Metal:

```bash
cd nca-mpsgraph
./check.sh accuracy --rollout-steps 100
./check.sh benchmark --rollout-steps 100 --benchmark-iterations 10 --warmup 2
```

PyTorch MPS:

```bash
cd nca-mpsgraph
uv run python benchmark_pytorch.py --steps 100 --iterations 10 --warmup 2 --device mps
```

## Accuracy Baseline

The direct Metal path stayed numerically stable through all successful variants:

- single-step forward: all tensors `PASS`
- single-step backward: all tensors `PASS`
- 100-step rollout: all steps `PASS`
- step 100 rollout max error: `1.901e-05`

This error pattern looks like normal float32 drift rather than a logic bug.

## Performance Timeline

### 1. Initial verified fast path

Approximate state after the first major cleanup pass:

- forward rollout was already faster than PyTorch
- single-step train-like path was still well behind

Measured shape:

- `forward_rollout_100`: about `64ms`
- `forward_backward_1`: about `3.8ms`
- `weight_grads_1`: about `2.48ms`

Main takeaway:

- forward orchestration was already in good shape
- the remaining gap was concentrated in weight-gradient kernels

### 2. Persistent buffers and forward-path cleanup

Changes:

- removed per-iteration allocations in the fast path
- kept hot buffers private/GPU-resident where possible
- reduced unnecessary command-buffer churn
- added specialized fused kernels for the forward masking path

Effect:

- major win on both rollout and train-like step
- forward path became faster than PyTorch
- backward gap remained mostly in weight grads

Representative result from that stage:

- `forward_rollout_100`: about `64ms`
- `forward_backward_1`: about `3.0ms`
- `weight_grads_1`: about `1.63ms`

### 3. Packed `HW-major` layout attempt

Idea:

- pack activations from `[B, C, HW]` to `[B, HW, C]`
- make weight-gradient reads contiguous and more GEMM-like

What was tried:

- explicit pack kernels
- weight-grad consumers rewritten to use packed buffers

Result:

- clear regression
- the extra full-buffer pack passes cost more than the consumer kernels saved

Representative result:

- `forward_backward_1`: about `7.4ms`
- `weight_grads_1`: about `5.8ms`

Conclusion:

- the idea is not necessarily wrong
- doing packing as separate global-memory passes was wrong for this shape regime

### 4. Direct tiled GEMM-like weight-grad attempt

Idea:

- mirror PyTorch’s logical formulation for linear backward weights:
  `grad_weight = grad_output^T @ input`
- replace custom row-reduction kernels with tiled matrix multiply style kernels

What was tried:

- explicit `[N, features]` flattening via pack kernels
- 16x16 tiled matmul-style weight-grad kernels
- a lighter scalar-tiled GEMM-style version before that

Result:

- both variants regressed badly

Representative regressions:

- scalar-tiled GEMM-like version:
  - `forward_backward_1`: about `10.6ms`
  - `weight_grads_1`: about `9.0ms`
- 16x16 pack-plus-matmul version:
  - `forward_backward_1`: about `8.0ms`
  - `weight_grads_1`: about `6.25ms`

Conclusion:

- the public PyTorch source correctly showed the dispatch/orchestration pattern
- my custom matmul microkernels were not competitive with the existing row-block kernels
- this does not disprove the flattened-matmul approach in principle
- it only shows that the attempted implementation was weaker than the existing specialized kernels

### 5. Row-block reduction family

This family consistently outperformed the pack-plus-matmul attempts.

Core idea:

- one threadgroup computes multiple adjacent input-channel gradients for one output row
- reuse the same `dOutput` values across several neighboring weights before reduction

Important variants:

#### Row-block x8

Configuration:

- 64 threads per group
- 8 adjacent input channels per output row

Result:

- strong baseline after reverting failed matmul attempts

Representative numbers:

- `forward_backward_1`: about `3.01ms`
- `weight_grads_1`: about `1.61ms`

#### `fc1` widened to 12 channels per block

Idea:

- `fc1` looked more reuse-limited than `fc2`
- widen only `fc1` first to reduce rereads of the same `d_fc1_raw` rows

Result:

- real improvement

Measured result:

- `forward_backward_1`: about `2.93ms`
- `weight_grads_1`: about `1.53ms`

Conclusion:

- reducing repeated output-row reads was the right lever
- register pressure had not yet overtaken the reuse benefit

#### `fc2` widened to 16 channels per block

Idea:

- apply the same reuse strategy to the smaller `fc2` weight-gradient kernel

Result:

- another real improvement

Measured result:

- `forward_backward_1`: about `2.81ms`
- `weight_grads_1`: about `1.42ms`

#### `fc1` widened again to 16 channels per block

Idea:

- remove one more pass over the same `d_fc1_raw` rows

Result:

- best result so far

Measured result:

- `forward_backward_1`: `2.60ms` GPU avg
- `weight_grads_1`: `1.23ms` GPU avg
- `forward_rollout_100`: `63.89ms` GPU avg

### 6. Two-step fused rollout attempt

Idea:

- fuse two deterministic forward NCA steps into one shared-memory kernel
- reduce per-step global writes and dispatch overhead
- test the cross-time-step fusion direction separately from backward/training changes

What was tried:

- added a rollout-only `2-step` fused kernel
- benchmarked it through a separate `forward_rollout_fused2_*` path
- kept the existing unfused fast rollout as the baseline

Correctness result:

- `rollout_steps=2`: `fused2_final max_err=2.384e-07 PASS`
- `rollout_steps=100`: `fused2_final max_err=4.567e-01 FAIL`

Performance result:

- baseline `forward_rollout_100`: `63.91ms` GPU avg
- fused `forward_rollout_fused2_100`: `288.15ms` GPU avg

Conclusion:

- this specific time-loop fusion attempt regressed badly
- it is accurate enough for a single fused chunk, but not stable enough over long rollouts
- it is also much slower than the current unfused fast path
- the likely reasons are heavy halo recomputation plus threshold-amplified float drift from the alive-mask path

### 7. Fully GPU-resident rollout loop

Idea:

- remove the remaining CPU-visible state handling from the timed rollout loop
- keep rollout state and scratch buffers in private storage
- reset the seed with GPU-to-GPU blits
- validate only the final state after `N` steps instead of reading every intermediate step

What was changed:

- rollout state buffers moved to private storage
- rollout scratch buffers moved to private storage
- one reusable encoded rollout helper added for the unfused fast path
- one final readback kept only for correctness checking

Correctness result:

- `rollout_final max_err=1.401e-05 PASS` at `100` steps

Performance result:

- `forward_rollout_resident_100`: `64.46ms` CPU avg, `63.99ms` GPU avg
- matching PyTorch MPS run on the same settings: `67.79ms`

Conclusion:

- the fully GPU-resident rollout loop is correct
- it does not materially change the rollout timing relative to the previous fast path
- this strongly suggests that CPU seed writes and final readback were not the main bottleneck in the rollout benchmark

## Current Best Known Result

Current direct Metal result:

- `forward_rollout_resident_100`: `63.99ms` GPU avg
- `forward_backward_1`: `2.60ms`
- `forward_only_1`: `0.65ms`
- `backward_core_1`: `0.72ms`
- `weight_grads_1`: `1.23ms`

Reference PyTorch MPS result on the same machine/settings:

- `forward_rollout_100`: `67.79ms`
- `forward_backward_1`: `2.51ms`

Interpretation:

- direct Metal is faster than PyTorch on the 100-step rollout
- direct Metal is within a few percent of PyTorch on the single-step train-like benchmark
- the remaining gap is very small and still concentrated in weight gradients

## Why 4090 Can Still Be 3-4x Faster

The current evidence points to hardware and backend maturity, not one missing Metal trick.

Main reasons:

- the model is small in parameter count, but not small in total work once `80-96` recurrent steps and backward are included
- an RTX 4090 has much higher raw compute throughput and much higher memory bandwidth than an M1 Max
- CUDA paths in PyTorch are more mature than MPS paths, especially for matmul/convolution-style work and launch/runtime behavior
- activation residency alone is probably not the explanation, because this model’s saved activations fit in a few gigabytes and can live on both devices

What the current measurements imply:

- dispatch overhead is not the main reason for the large 4090 gap
- CPU readback is not the main reason for the large 4090 gap either, because the fully GPU-resident rollout changed almost nothing
- the remaining difference is much more likely to come from raw hardware advantage plus stronger CUDA backend kernels than from a simple Metal orchestration mistake

## What Clearly Regressed

These did not help in the tested form:

- explicit pack/transpose passes to `HW-major` layouts
- half-converted dual-layout flows where the producer/consumer pair was incomplete
- generic barrier-heavy tiled matmul kernels
- the current shared-memory `2-step` rollout fusion kernel

Reason they likely regressed:

- they added full-memory passes or stronger synchronization without enough compensating reuse

## What Clearly Helped

- persistent scratch buffers
- keeping the fast path GPU-resident
- specialized forward fusions for residual/mask operations
- layer-specific, shape-specific weight-gradient kernels
- widening row-block kernels to reuse more `dOutput` work per threadgroup

## Likely Next Ideas

Promising:

- direct dual-write producer kernels that emit both channel-major and consumer-friendly flattened layouts without separate pack kernels
- further layer-specific tuning rather than forcing the same strategy on `fc1` and `fc2`
- more aggressive custom fusion only after establishing per-step parity

Less promising in the already-tested form:

- standalone pack passes
- generic custom GEMM kernels without stronger microkernel design

## Current Best Kernel Notes

Best current weight-gradient configuration:

- `fc1_weight_grad_tiled`: 16-channel block per output row
- `fc2_weight_grad_tiled`: 16-channel block per output row
- dispatch remains one threadgroup per `(output_row, input_channel_block)`

Relevant files:

- `nca-mpsgraph/Sources/NCATrainer/nca_kernels.metal`
- `nca-mpsgraph/Sources/NCATrainer/main.swift`
