"""Generate reference NCA outputs from PyTorch for validating the WebGPU shader.

Runs deterministic steps (fire_rate=1.0, no stochastic mask) so the WebGPU
implementation can be compared numerically.

Usage:
    uv run python nca-webgpu/generate_reference.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from nca_model import NCAModel
from train import make_seed, CHANNEL_N, GRID_SIZE

WEIGHTS_BIN = "output/A/weights.bin"
CHECKPOINT = "output/A/checkpoint_final.pt"
OUTPUT_DIR = "nca-webgpu/reference"
N_STEPS = 10


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")  # CPU for exact reproducibility

    # Load model
    model = NCAModel(CHANNEL_N, 128, fire_rate=1.0).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Seed state
    seed = torch.from_numpy(make_seed()).to(device)  # [1, 16, 72, 72]

    # Save seed as reference
    save_state(seed, os.path.join(OUTPUT_DIR, "step_000.bin"))

    # Run N steps deterministically (fire_rate=1.0 → all cells update)
    x = seed.clone()
    with torch.no_grad():
        for i in range(1, N_STEPS + 1):
            x = model(x, fire_rate=1.0)
            save_state(x, os.path.join(OUTPUT_DIR, f"step_{i:03d}.bin"))
            if i <= 3 or i == N_STEPS:
                print_stats(x, i)

    print(f"\nSaved {N_STEPS + 1} reference states to {OUTPUT_DIR}/")
    print(f"Weights: {WEIGHTS_BIN} ({os.path.getsize(WEIGHTS_BIN)} bytes)")


def save_state(x: torch.Tensor, path: str):
    """Save state tensor as flat float32 binary. Layout: (y * W + x) * C + c."""
    # x is [1, C, H, W] in PyTorch (NCHW).
    # WebGPU expects [H, W, C] flattened as (y * W + x) * C + c.
    arr = x[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [H, W, C]
    arr.astype(np.float32).tofile(path)


def print_stats(x: torch.Tensor, step: int):
    rgba = x[0, :4].cpu().numpy()  # [4, H, W]
    alpha = rgba[3]
    alive = (alpha > 0.1).sum()
    print(
        f"  Step {step:3d}: "
        f"alpha range [{alpha.min():.4f}, {alpha.max():.4f}], "
        f"alive cells: {alive}, "
        f"RGB mean (alive): {rgba[:3][:, alpha > 0.1].mean():.4f}"
        if alive > 0
        else f"  Step {step:3d}: no alive cells"
    )


if __name__ == "__main__":
    main()
