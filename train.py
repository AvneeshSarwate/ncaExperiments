"""Training script for Neural Cellular Automata.

Implements pool-based training with damage repair from:
    Mordvintsev et al., "Growing Neural Cellular Automata" (Distill, 2020)

Usage:
    # Train with a provided 40x40 RGBA target image:
    uv run python train.py --target targets/R.png

    # Train with an auto-generated test character:
    uv run python train.py --char R

    # Resume from checkpoint:
    uv run python train.py --target targets/R.png --resume output/R/checkpoint_004000.pt
"""

import argparse
import base64
import json
import os
import struct
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca_model import NCAModel

# --- Hyperparameters (from the paper) ---
CHANNEL_N = 16
HIDDEN_N = 128
FIRE_RATE = 0.5
POOL_SIZE = 1024
BATCH_SIZE = 8
DAMAGE_N = 3  # number of best samples to damage each step
TARGET_SIZE = 40
TARGET_PADDING = 16
GRID_SIZE = TARGET_SIZE + 2 * TARGET_PADDING  # 72
LR_INITIAL = 2e-3
LR_DECAY_STEP = 2000
LR_DECAY_FACTOR = 0.1
TRAIN_STEPS = 8000
MIN_NCA_STEPS = 64
MAX_NCA_STEPS = 96


# --- Sample Pool ---


class SamplePool:
    """Pool of NCA grid states for stable training.

    Maintains a fixed-size pool of partially-grown states. Each training step
    samples a batch, runs the NCA forward, and writes the results back. This
    prevents the model from only ever seeing seed states and forces it to
    maintain patterns over long horizons.
    """

    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = list(slots.keys())
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, replace=False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool(**batch, _parent=self, _parent_idx=idx)

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


# --- Helpers ---


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_seed(channel_n=CHANNEL_N, grid_size=GRID_SIZE):
    """Seed state: all zeros, center pixel has alpha + hidden channels = 1.0."""
    seed = np.zeros([1, channel_n, grid_size, grid_size], np.float32)
    mid = grid_size // 2
    seed[0, 3:, mid, mid] = 1.0  # alpha=1, hidden=1, RGB=0
    return seed


def make_circle_masks(n, h, w):
    """Random circular damage masks. Returns [n, h, w] float32 (1 inside circle)."""
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, (2, n, 1, 1))
    r = np.random.uniform(0.1, 0.4, (n, 1, 1))
    xc = (x - center[0]) / r
    yc = (y - center[1]) / r
    return (xc * xc + yc * yc < 1.0).astype(np.float32)


def load_target(path, target_size=TARGET_SIZE, padding=TARGET_PADDING):
    """Load a target image, resize to target_size, pad to grid_size. Returns [1,4,H,W]."""
    img = Image.open(path).convert("RGBA")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    target = np.array(img, dtype=np.float32) / 255.0  # [H, W, 4]
    padded = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
    return padded.transpose(2, 0, 1)[np.newaxis]  # [1, 4, H, W]


def to_rgba(x):
    """Extract RGBA channels (first 4) from state tensor."""
    return x[:, :4]


def loss_fn(x, target):
    """Per-sample MSE on RGBA channels. Returns [B] tensor."""
    return F.mse_loss(to_rgba(x), target, reduction="none").mean(dim=[1, 2, 3])


def normalize_grads(model):
    """Per-variable L2 gradient normalization (from the paper, not standard clipping)."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.copy_(p.grad.data / (p.grad.data.norm() + 1e-8))


def generate_test_target(char="R", output_path=None):
    """Generate a bold character on transparent background for testing."""
    img = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try system fonts for a bold, thick stroke
    font = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 30)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Center the character
    bbox = draw.textbbox((0, 0), char, font=font)
    tx = (TARGET_SIZE - (bbox[2] - bbox[0])) // 2 - bbox[0]
    ty = (TARGET_SIZE - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((tx, ty), char, fill=(255, 255, 255, 255), font=font)

    if output_path:
        img.save(output_path)
    return img


# --- Visualization ---


def get_bg_color(target_np):
    """Compute background color as the RGB negative of the target's stroke color."""
    alpha = target_np[3]
    mask = alpha > 0.5
    if mask.any():
        fg = target_np[:3, mask].mean(axis=1)  # [3]
        return 1.0 - fg
    return np.array([0.0, 0.0, 0.0])


def save_visualization(x, target, step, vis_dir):
    """Save batch states as a grid image."""
    rgba = to_rgba(x).detach().cpu().clamp(0, 1).numpy()
    target_np = target[0].cpu().numpy()  # [4, H, W]
    bg = get_bg_color(target_np).reshape(1, 1, 3)

    n = min(len(rgba), BATCH_SIZE)
    cols = min(n + 1, 5)
    rows = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx == 0:
            rgb = target_np[:3].transpose(1, 2, 0)
            alpha = target_np[3:4].transpose(1, 2, 0)
            ax.imshow(rgb * alpha + bg * (1 - alpha), vmin=0, vmax=1)
            ax.set_title("Target", fontsize=8)
        elif idx - 1 < n:
            img = rgba[idx - 1]
            rgb = img[:3].transpose(1, 2, 0)
            alpha = img[3:4].transpose(1, 2, 0)
            ax.imshow(rgb * alpha + bg * (1 - alpha), vmin=0, vmax=1)
            ax.set_title(f"Sample {idx - 1}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"step_{step:06d}.png"), dpi=100)
    plt.close()


# --- Weight Export ---


def export_weights(model, output_dir):
    """Export model weights for WebGPU inference.

    Produces:
      - weights.json: JSON with base64-encoded float32 weight blobs + shapes
      - weights.bin: raw concatenated float32 arrays (conv1.weight, conv1.bias, conv2.weight)
    """
    c1w = model.fc1.weight.detach().cpu().numpy()  # [128, 48, 1, 1]
    c1b = model.fc1.bias.detach().cpu().numpy()  # [128]
    c2w = model.fc2.weight.detach().cpu().numpy()  # [16, 128, 1, 1]

    # JSON with base64
    def to_b64(arr):
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

    weights_json = {
        "channel_n": model.channel_n,
        "hidden_n": model.hidden_n,
        "conv1_weight": {"shape": list(c1w.shape), "data_b64": to_b64(c1w)},
        "conv1_bias": {"shape": list(c1b.shape), "data_b64": to_b64(c1b)},
        "conv2_weight": {"shape": list(c2w.shape), "data_b64": to_b64(c2w)},
    }

    json_path = os.path.join(output_dir, "weights.json")
    with open(json_path, "w") as f:
        json.dump(weights_json, f, indent=2)

    # Raw binary: conv1.weight | conv1.bias | conv2.weight
    bin_path = os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        f.write(c1w.astype(np.float32).tobytes())
        f.write(c1b.astype(np.float32).tobytes())
        f.write(c2w.astype(np.float32).tobytes())

    total_floats = c1w.size + c1b.size + c2w.size
    total_bytes = total_floats * 4
    print(f"Exported weights: {total_floats:,} floats ({total_bytes:,} bytes)")
    print(f"  {json_path}")
    print(f"  {bin_path}")


# --- Training ---


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Output directories
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate target
    if args.target:
        target_np = load_target(args.target)
        print(f"Loaded target: {args.target}")
    else:
        char = args.char or "R"
        target_path = output_dir / f"target_{char}.png"
        generate_test_target(char, str(target_path))
        target_np = load_target(str(target_path))
        print(f"Generated test target for '{char}' at {target_path}")

    target = torch.from_numpy(target_np).to(device)  # [1, 4, H, W]
    target_batch = target.expand(BATCH_SIZE, -1, -1, -1)  # [B, 4, H, W]

    # Model
    model = NCAModel(CHANNEL_N, HIDDEN_N, FIRE_RATE).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer + LR schedule (2e-3 for steps 0-1999, 2e-4 for steps 2000+)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 if step < LR_DECAY_STEP else LR_DECAY_FACTOR,
    )

    # Seed and pool
    seed = make_seed()
    pool = SamplePool(x=np.repeat(seed, POOL_SIZE, axis=0))

    # Resume
    start_step = 0
    losses = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if "pool" in ckpt:
            pool = SamplePool(x=ckpt["pool"])
        if "losses" in ckpt:
            losses = ckpt["losses"]
        print(f"Resumed from step {start_step}")

    # Training loop
    t0 = time.time()
    model.train()

    for step in range(start_step, args.steps):
        # 1. Sample batch from pool
        batch = pool.sample(BATCH_SIZE)
        x0 = batch.x.copy()

        # 2. Sort by loss descending (highest loss first)
        with torch.no_grad():
            x0_t = torch.from_numpy(x0).to(device)
            sample_losses = loss_fn(x0_t, target_batch).cpu().numpy()
        loss_rank = sample_losses.argsort()[::-1].copy()
        x0 = x0[loss_rank]

        # 3. Replace highest-loss sample with fresh seed
        x0[:1] = seed

        # 4. Damage the best (lowest-loss) samples with circular masks
        if DAMAGE_N > 0:
            masks = make_circle_masks(DAMAGE_N, GRID_SIZE, GRID_SIZE)
            damage = 1.0 - masks[:, np.newaxis, :, :]  # [N, 1, H, W]
            x0[-DAMAGE_N:] *= damage

        # 5. Forward: run NCA for random [64, 96) steps
        x = torch.from_numpy(x0).to(device)
        iter_n = np.random.randint(MIN_NCA_STEPS, MAX_NCA_STEPS)

        for _ in range(iter_n):
            x = model(x)

        # 6. Loss + backward
        loss_per_sample = loss_fn(x, target_batch)
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        normalize_grads(model)
        optimizer.step()
        scheduler.step()

        # 7. Write states back to pool (undo the sort permutation)
        batch.x[loss_rank] = x.detach().cpu().numpy()
        batch.commit()

        # --- Logging ---
        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step:5d}/{args.steps}"
                f"  loss={loss_val:.6f}"
                f"  lr={lr:.1e}"
                f"  ({elapsed:.0f}s)"
            )

        if step % 500 == 0:
            save_visualization(x, target, step, str(vis_dir))

        if step > 0 and step % 2000 == 0:
            ckpt_path = str(output_dir / f"checkpoint_{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "pool": pool.x,
                    "losses": losses,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Final outputs ---
    total_time = time.time() - t0
    print(f"\nTraining complete. {args.steps} steps in {total_time:.0f}s")

    # Final checkpoint (model-only, no pool — smaller file)
    torch.save(
        {"step": args.steps, "model": model.state_dict(), "losses": losses},
        str(output_dir / "checkpoint_final.pt"),
    )

    # Export weights for WebGPU
    export_weights(model, str(output_dir))

    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "loss.png"), dpi=100)
    plt.close()

    # Final visualization
    save_visualization(x, target, args.steps, str(vis_dir))

    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCA for a target character")
    parser.add_argument("--target", type=str, help="Path to 40x40 RGBA target PNG")
    parser.add_argument(
        "--char",
        type=str,
        default="R",
        help="Character to generate as test target (if --target not given)",
    )
    parser.add_argument(
        "--steps", type=int, default=TRAIN_STEPS, help="Training steps (default: 8000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/<char>)",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.output_dir is None:
        char_name = args.char or "R"
        if args.target:
            char_name = Path(args.target).stem
        args.output_dir = f"output/{char_name}"

    train(args)
