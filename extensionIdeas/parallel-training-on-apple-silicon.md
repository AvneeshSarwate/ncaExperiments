# Parallel Training on Apple Silicon

## Problem

Each NCA model is tiny (~8,300 params, 72x72 grid) and heavily underutilizes the M1 Max 32-core GPU. Training one character takes ~15 min. Training 26 characters sequentially takes ~6.5 hours. Multi-process parallelism was attempted but made things 20% slower due to MPS dispatch overhead contention.

## Why multi-process fails

The bottleneck is CPU-GPU dispatch overhead, not GPU compute. Each NCA forward step fires ~12-15 MPS kernel dispatches (depthwise conv, two 1x1 convs, ReLU, max_pool, random mask, element-wise ops). At 64-96 sequential steps per training iteration, that's 640-1440 dispatches, each with ~20us overhead. Two processes double the dispatch contention on a single Metal command queue. Measured: solo = 8.9 steps/s, 2 processes = 3.6 steps/s each (7.2 total), which is slower.

Apple Silicon has only 2 hardware concurrency slots across command queues (per Philip Turner's metal-benchmarks). The PyTorch MPS software overhead exceeds the hardware parallelism benefit.

## Approaches worth investigating

### 1. Grouped convolution trick (stays in PyTorch)

All NCA models have identical architecture but different weights. Train N models as one model using `groups=N`:

- Stack N models' conv weights into grouped conv layers
- Concatenate N inputs along channel dim: `[B, 16*N, 72, 72]`
- `F.conv2d(..., groups=N)` applies N independent convolutions in one kernel launch
- Collapses N*K kernel launches into K launches

Engineering needed: N independent pools, per-model loss computation, weight stacking/unstacking, per-model optimizer states. Expected ~3-4x throughput for N=4.

### 2. Reduce CPU-GPU roundtrips

The training loop forces synchronization each step by moving data to CPU:
- Pool is numpy on CPU (340MB) -- could live on GPU
- Loss ranking uses `.cpu().numpy()` + `argsort` -- could use `torch.argsort` on GPU
- Damage masks generated with numpy -- could use torch on GPU

Estimated 10-30% improvement. Easy to implement.

### 3. JAX + jax.lax.scan (biggest potential)

JAX can compile the entire 64-96 step NCA loop into a single fused dispatch via `lax.scan`. The CAX library (Cellular Automata Accelerated in JAX, ICLR 2025 Oral) claims up to 2000x speedup for cellular automata. `jax-metal` provides Apple Silicon GPU support via PJRT/OpenXLA. The NCA model is simple enough to port in a day.

### 4. MLX + mx.compile (Apple-native)

MLX's JIT compiler (`mx.compile`) fuses entire functions into single GPU kernels, eliminating per-op dispatch overhead. However, M1 Max conv2d performance in MLX is reportedly 3-6x worse than PyTorch MPS (improves on M3+). Worth benchmarking.

### 5. torch.func.vmap ensemble (conceptually elegant, fragile)

PyTorch's `torch.func.stack_module_state` + `vmap` + `functional_call` can vectorize N models into a single batched call. However, MPS support is untested and there are known issues with optimizer integration even on CUDA (gradients zeroing after `optimizer.step()`).

## What doesn't work

- **torch.compile on MPS**: backward pass crashes (PyTorch issue #161905), not usable for training as of PyTorch 2.11
- **Multi-process parallelism**: dispatch overhead kills throughput
- **Metal streams/concurrent dispatch**: not accessible from PyTorch MPS
- **Increasing batch size**: changes the training math (loss averaging, damage ratio, seed replacement ratio) and would require retuning all hyperparameters

## Current recommendation

## Ruled out: JAX and MLX ports (March 2026)

### JAX + jax-metal
- **jax-metal is abandoned by Apple.** Last release 0.1.1 (early 2024), incompatible with JAX > 0.5.0.
- Error: `UNIMPLEMENTED: default_memory_space is not supported` on any modern JAX.
- jax-mps (community alternative using MLX backend) is too early — most StableHLO ops unimplemented.
- Even if it worked, no confirmation that `lax.scan` functions on the Metal backend.

### MLX
- **conv2d is 2-4x slower than PyTorch MPS on M1 Max** (mlx-benchmark project, confirmed in MLX issue #1409). No 2D conv improvements in any release through March 2025.
- **mx.compile only fuses element-wise ops** (ReLU, add, mul). Conv2d, max_pool, matmul remain as separate kernel dispatches. The fusion does NOT help with the dispatch overhead problem.
- **No lax.scan equivalent.** Feature request #1441 open since September 2024 with no implementation. Python loops are unrolled at trace time into a flat graph, still dispatching each conv individually.
- **Dynamic loop counts cause recompilation.** The 64-96 random step range would generate 32 compiled variants.

### The fundamental issue
No framework on Apple Silicon can compile a sequential loop of convolutions into a fused kernel. This capability exists only on CUDA (via XLA/Inductor/Triton). The Apple GPU ecosystem lacks scan primitives and cross-operation kernel fusion for non-element-wise ops.

## Still worth investigating

### PyTorch CPU with torch.compile
For a model this tiny (8K params, 72x72 grid), CPU compute is trivial. The entire bottleneck is GPU dispatch overhead, which CPU doesn't have. `torch.compile` with the Inductor backend CAN fuse operations on CPU. Could be competitive or faster than MPS for this specific workload. Needs benchmarking.

### Custom Metal shader for training
Write the entire NCA forward pass (perceive + update + mask) as a single Metal compute kernel, reducing each NCA step to 1 dispatch instead of ~12. Combine with PyTorch via custom MPS ops. High engineering effort but would directly solve the dispatch bottleneck.

## CONFIRMED: MPSGraph native training (March 2026 PoC)

A Swift proof-of-concept in `nca-mpsgraph/` validated:

1. **Forward pass parity**: MPSGraph NCA matches PyTorch within 2.86e-6 max error over 10 steps (float32 rounding only) when using hard threshold alive mask
2. **Single-step gradients**: All weight tensors (fc1_w, fc1_b, fc2_w) receive non-zero gradients via `graph.gradients()`
3. **For-loop + gradients (BPTT)**: `graph.for(numberOfIterations:...)` with `graph.gradients()` through the loop body **works** — all gradients non-zero through 4 NCA iterations

**Key finding**: MPSGraph's `greaterThan` has no gradient implementation (assertion failure). The fix is using `sigmoid((maxAlpha - 0.1) * 100)` as a smooth differentiable approximation of the alive mask. With steepness=100, this is nearly identical to a hard threshold.

**What this enables**: The entire training step (64-96 NCA forward passes + backprop + weight update) can be a SINGLE MPSGraph execution — zero CPU-GPU synchronization per NCA step. This directly eliminates the ~1000+ kernel dispatch bottleneck.

**Remaining work to build the full trainer**:
- Pool management (sample, sort by loss, seed replacement, damage)
- Stochastic fire mask (MPSGraph has random ops)
- Adam optimizer (MPSGraph has built-in Adam)
- LR schedule (piecewise constant)
- Weight I/O (load/save, export for WebGPU)
- Benchmark vs PyTorch MPS to quantify the speedup

## Current recommendation

For < 26 characters, just run sequentially on PyTorch MPS. For full alphabet or iterative training, the MPSGraph native path is confirmed viable and should give significant speedup on Apple Silicon. On NVIDIA GPUs (RunPod), use PyTorch CUDA with NVIDIA MPS daemon for multi-process parallelism.
