"""Benchmark PyTorch rollout speed against the direct-Metal verifier."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_model import NCAModel
from train import CHANNEL_N, GRID_SIZE, HIDDEN_N, TARGET_PADDING, find_seed_position, load_target, make_seed


ROOT = Path(__file__).resolve().parent
WEIGHTS_BIN = ROOT.parent / "output" / "A" / "weights.bin"
TARGET = ROOT.parent / "images" / "A.png"
BATCH = 8
PERC = CHANNEL_N * 3


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(device: torch.device) -> NCAModel:
    model = NCAModel(CHANNEL_N, HIDDEN_N, fire_rate=1.0).to(device)
    weights = np.fromfile(WEIGHTS_BIN, dtype=np.float32)

    fc1_w = torch.from_numpy(weights[: HIDDEN_N * PERC].reshape(HIDDEN_N, PERC, 1, 1)).to(device)
    fc1_b = torch.from_numpy(weights[HIDDEN_N * PERC : HIDDEN_N * PERC + HIDDEN_N]).to(device)
    fc2_w = torch.from_numpy(
        weights[HIDDEN_N * PERC + HIDDEN_N : HIDDEN_N * PERC + HIDDEN_N + CHANNEL_N * HIDDEN_N].reshape(
            CHANNEL_N, HIDDEN_N, 1, 1
        )
    ).to(device)

    with torch.no_grad():
        model.fc1.weight.copy_(fc1_w)
        model.fc1.bias.copy_(fc1_b)
        model.fc2.weight.copy_(fc2_w)

    return model


def make_seed_batch(device: torch.device) -> torch.Tensor:
    target_np = load_target(str(TARGET))
    seed_y, seed_x = find_seed_position(target_np, TARGET_PADDING)
    seed = make_seed(CHANNEL_N, GRID_SIZE, seed_y, seed_x).astype(np.float32)
    batch = np.repeat(seed, BATCH, axis=0)
    return torch.from_numpy(batch).to(device)


def benchmark_forward_rollout(model: NCAModel, seed: torch.Tensor, steps: int, iterations: int, warmup: int) -> tuple[float, float]:
    times = []
    with torch.no_grad():
        for idx in range(warmup + iterations):
            x = seed.clone()
            synchronize(x.device)
            t0 = time.perf_counter()
            for _ in range(steps):
                x = model(x, fire_rate=1.0)
            synchronize(x.device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if idx >= warmup:
                times.append(dt_ms)
    avg_ms = float(np.mean(times))
    steps_per_sec = steps * 1000.0 / avg_ms
    return avg_ms, steps_per_sec


def benchmark_forward_backward(model: NCAModel, seed: torch.Tensor, iterations: int, warmup: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    dout_np = rng.standard_normal(seed.shape, dtype=np.float32)
    d_output = torch.from_numpy(dout_np).to(seed.device)

    times = []
    for idx in range(warmup + iterations):
        x = seed.clone().requires_grad_(True)
        for param in model.parameters():
            param.grad = None

        synchronize(seed.device)
        t0 = time.perf_counter()
        output = model(x, fire_rate=1.0)
        (output * d_output).sum().backward()
        synchronize(seed.device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= warmup:
            times.append(dt_ms)

    avg_ms = float(np.mean(times))
    steps_per_sec = 1000.0 / avg_ms
    return avg_ms, steps_per_sec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(device).eval()
    seed = make_seed_batch(device)

    rollout_ms, rollout_steps_per_sec = benchmark_forward_rollout(model, seed, args.steps, args.iterations, args.warmup)
    fb_ms, fb_steps_per_sec = benchmark_forward_backward(model, seed, args.iterations, args.warmup)

    print(f"Device: {device}")
    print(f"forward_rollout_{args.steps}: avg={rollout_ms:.2f}ms throughput={rollout_steps_per_sec:.2f}/s")
    print(f"forward_backward_1: avg={fb_ms:.2f}ms throughput={fb_steps_per_sec:.2f}/s")


if __name__ == "__main__":
    main()
