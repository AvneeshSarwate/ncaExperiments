"""Generate NCHW rollout references from PyTorch for validating the Metal implementation."""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from nca_model import NCAModel
from train import load_target, find_seed_position, CHANNEL_N, GRID_SIZE, TARGET_PADDING

TARGET = "../images/A.png"
OUTPUT_DIR = "reference"


def save_nchw(x, path):
    """Save [1, C, H, W] tensor as flat float32 in NCHW order."""
    x[0].contiguous().cpu().numpy().astype(np.float32).tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    model = NCAModel(CHANNEL_N, 128, fire_rate=1.0).to(device)
    ckpt = torch.load("../output/A/checkpoint_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Find seed position (same logic as training)
    target_np = load_target(TARGET)
    seed_y, seed_x = find_seed_position(target_np, TARGET_PADDING)
    print(f"Seed position: ({seed_x}, {seed_y})")

    # Create seed in NCHW
    seed = torch.zeros(1, CHANNEL_N, GRID_SIZE, GRID_SIZE)
    seed[0, 3:, seed_y, seed_x] = 1.0

    save_nchw(seed, os.path.join(OUTPUT_DIR, "step_000.bin"))

    x = seed.clone()
    with torch.no_grad():
        for i in range(1, args.steps + 1):
            x = model(x, fire_rate=1.0)
            save_nchw(x, os.path.join(OUTPUT_DIR, f"step_{i:03d}.bin"))
            alpha = x[0, 3].numpy()
            alive = (alpha > 0.1).sum()
            print(f"  Step {i:3d}: alive={alive}, alpha=[{alpha.min():.4f}, {alpha.max():.4f}]")

    # Also save seed position for the Swift code
    with open(os.path.join(OUTPUT_DIR, "seed_pos.txt"), "w") as f:
        f.write(f"{seed_y} {seed_x}\n")

    print(f"\nSaved {args.steps + 1} NCHW reference states to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
