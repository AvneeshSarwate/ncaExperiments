"""Generate single-step forward/backward PyTorch references for the Metal verifier."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_model import NCAModel
from train import CHANNEL_N, GRID_SIZE, HIDDEN_N, TARGET_PADDING, find_seed_position, load_target, make_seed


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reference_ops"
WEIGHTS_BIN = ROOT.parent / "output" / "A" / "weights.bin"
TARGET = ROOT.parent / "images" / "A.png"

BATCH = 8
PERC = CHANNEL_N * 3


def save_nchw(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x.detach().cpu().contiguous().numpy().astype(np.float32).tofile(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    model = NCAModel(CHANNEL_N, HIDDEN_N, fire_rate=1.0).to(device)
    weights = np.fromfile(WEIGHTS_BIN, dtype=np.float32)

    fc1_w = torch.from_numpy(weights[: HIDDEN_N * PERC].reshape(HIDDEN_N, PERC, 1, 1)).to(device)
    fc1_b = torch.from_numpy(weights[HIDDEN_N * PERC : HIDDEN_N * PERC + HIDDEN_N]).to(device)
    fc2_w = torch.from_numpy(
        weights[HIDDEN_N * PERC + HIDDEN_N : HIDDEN_N * PERC + HIDDEN_N + CHANNEL_N * HIDDEN_N].reshape(
            CHANNEL_N, HIDDEN_N, 1, 1
        )
    ).to(device)

    fc1_w.requires_grad_(True)
    fc1_b.requires_grad_(True)
    fc2_w.requires_grad_(True)

    target_np = load_target(str(TARGET))
    seed_y, seed_x = find_seed_position(target_np, TARGET_PADDING)
    seed = make_seed(CHANNEL_N, GRID_SIZE, seed_y, seed_x)
    x_np = np.repeat(seed, BATCH, axis=0).astype(np.float32)
    x = torch.from_numpy(x_np).to(device).requires_grad_(True)

    perc = F.conv2d(x, model.perception_kernel, padding=1, groups=CHANNEL_N)
    fc1_out = F.conv2d(perc, fc1_w, fc1_b)
    hidden = F.relu(fc1_out)
    delta = F.conv2d(hidden, fc2_w)

    max_alpha_pre = F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1)
    pre_mask = (max_alpha_pre > 0.1).float()

    fire_mask = torch.ones_like(pre_mask)
    updated = x + delta * fire_mask

    max_alpha_post = F.max_pool2d(updated[:, 3:4], 3, stride=1, padding=1)
    post_mask = (max_alpha_post > 0.1).float()
    life_mask = pre_mask * post_mask
    output = updated * life_mask

    for tensor in (perc, fc1_out, hidden, delta, updated, output):
        tensor.retain_grad()

    rng = np.random.default_rng(0)
    d_output_np = rng.standard_normal(output.shape, dtype=np.float32)
    d_output = torch.from_numpy(d_output_np).to(device)

    (output * d_output).sum().backward()

    save_nchw(x, OUTPUT_DIR / "input.bin")
    save_nchw(perc, OUTPUT_DIR / "perc.bin")
    save_nchw(fc1_out, OUTPUT_DIR / "fc1_out.bin")
    save_nchw(hidden, OUTPUT_DIR / "hidden.bin")
    save_nchw(delta, OUTPUT_DIR / "delta.bin")
    save_nchw(max_alpha_pre, OUTPUT_DIR / "max_alpha_pre.bin")
    save_nchw(pre_mask, OUTPUT_DIR / "pre_mask.bin")
    save_nchw(updated, OUTPUT_DIR / "updated.bin")
    save_nchw(max_alpha_post, OUTPUT_DIR / "max_alpha_post.bin")
    save_nchw(post_mask, OUTPUT_DIR / "post_mask.bin")
    save_nchw(life_mask, OUTPUT_DIR / "life_mask.bin")
    save_nchw(d_output, OUTPUT_DIR / "d_output.bin")
    save_nchw(updated.grad, OUTPUT_DIR / "d_updated.bin")
    save_nchw(delta.grad, OUTPUT_DIR / "d_delta.bin")
    save_nchw(hidden.grad, OUTPUT_DIR / "d_hidden.bin")
    save_nchw(fc1_out.grad, OUTPUT_DIR / "d_fc1_raw.bin")
    save_nchw(perc.grad, OUTPUT_DIR / "d_perc.bin")
    save_nchw(x.grad, OUTPUT_DIR / "d_state.bin")
    save_nchw(fc1_w.grad.reshape(HIDDEN_N, PERC), OUTPUT_DIR / "d_fc1_w.bin")
    save_nchw(fc1_b.grad, OUTPUT_DIR / "d_fc1_b.bin")
    save_nchw(fc2_w.grad.reshape(CHANNEL_N, HIDDEN_N), OUTPUT_DIR / "d_fc2_w.bin")

    print(f"Saved forward/backward op references to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
