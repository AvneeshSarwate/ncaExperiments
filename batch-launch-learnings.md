# Batch Training Learnings — RunPod RTX 4090

## Environment
- RunPod pod with RTX 4090 (24 GB VRAM)
- 20 GB network volume quota on `/workspace`
- NCA model: ~8,300 params, 72x72 grid, 16 channels

## Memory bottleneck

The model is tiny (33 KB weights), but backprop through 64-96 sequential NCA steps stores intermediate activations at every step. The fc1 output `[8, 128, 72, 72]` alone is ~21 MB per step. Over 80 steps, activation memory dominates at ~2.6 GB per model (envelope estimate) / ~3.4 GB per process (measured, including PyTorch allocator overhead and variable step counts).

## Parallel process count

- **6 processes** is the safe max on 24 GB VRAM at batch size 8.
- 7 processes OOM'd — all 7 crashed simultaneously on the first NCA step that pushed past the limit.
- NVIDIA MPS daemon was running but didn't meaningfully help. The bottleneck is VRAM capacity, not dispatch contention (unlike Apple Silicon where dispatch overhead was the chokepoint).

## GPU utilization

- Single process: 0% utilization reported by `nvidia-smi` — kernels are too short-lived for the sampling interval to catch.
- 6 parallel processes: 100% utilization, ~22 GB VRAM. This is the correct regime — naive multi-process parallelism works because we're now VRAM-bound rather than dispatch-bound.

## CUDA vs Apple Silicon (M1 Max)

On Metal, the bottleneck was kernel dispatch overhead (~20us per op, 12 ops per NCA step, single Metal command queue with 2 concurrency slots). Multi-process parallelism made contention *worse*. On CUDA, dispatch is fast enough (~5us) that the bottleneck shifts to VRAM — which is the "correct" bottleneck for parallelism. Same model, same loop unrolling, fundamentally different scaling behavior.

Single character: ~5 min on 4090 vs ~15-20 min on M1 Max.

## Disk quota

Checkpoints include the full training pool (1024 x 16 x 72 x 72 float32 = ~340 MB each). With 6 parallel jobs checkpointing every 2000 steps, disk fills fast:
- Original: 2 checkpoints per model x 6 models x ~380 MB = ~4.5 GB of checkpoints alone.
- Fix: delete previous checkpoint *before* writing the new one (not after), and delete the final checkpoint on training completion. This keeps max concurrent checkpoint disk at ~2.3 GB (one per model).
- Also: the `uv` package cache (`/workspace/.persist/uv-cache`) was 8 GB. Clearing it with `uv cache clean` freed enough space. Packages remain installed in `.venv`.

### Disk corruption

When `torch.save` hits a full disk, it can write a truncated file silently (no exception until close). Worse, if the training script itself (`train.py`) is on the same filesystem and gets written to by an editor during a disk-full event, it can be corrupted with null bytes. Always `git checkout -- train.py` to restore if this happens.

Truncated checkpoints are identifiable by size: valid ones are ~370-400 MB, corrupt ones are 0-192 MB.

## Python stdout buffering

When redirecting output to a log file (`> /tmp/train_X.log`), Python buffers all stdout. Logs appear empty until the process exits. Fix: use `python -u` for unbuffered output so you can monitor progress mid-run.

## Working directory

Background shell commands (`&`) inherit the current working directory of the shell session, not the directory you might expect from the conversation context. Always `cd /workspace/ncaExperiments &&` before launching, or use absolute paths.

## Practical throughput

- 6 characters from step 0: ~15 min wall time
- 14 characters (A-N): 2 rounds (6 + 5), ~30 min total (with some time lost to disk issues and restarts)
- Projected 26 characters: ceil(26/6) = 5 rounds, ~75 min if clean. Closer to ~60 min if some rounds have fewer than 6.
