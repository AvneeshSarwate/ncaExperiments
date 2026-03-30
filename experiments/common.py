"""Shared utilities for NCA experiments.

Extracted from train.py with parameterized interfaces for reuse across
different experiment configurations.
"""

import base64
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# --- Default hyperparameters (from the paper) ---

DEFAULTS = {
    "channel_n": 16,
    "hidden_n": 128,
    "fire_rate": 0.5,
    "pool_size": 1024,
    "batch_size": 8,
    "damage_n": 3,
    "target_size": 40,
    "target_padding": 16,
    "lr_initial": 2e-3,
    "lr_decay_step": 2000,
    "lr_decay_factor": 0.1,
    "steps": 8000,
    "min_nca_steps": 64,
    "max_nca_steps": 96,
}


# --- Device ---


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def grid_size(target_size=DEFAULTS["target_size"], padding=DEFAULTS["target_padding"]):
    return target_size + 2 * padding


# --- Sample Pool ---


class SamplePool:
    """Pool of NCA grid states for stable training."""

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


def find_seed_position(target_np, padding):
    """Find the best seed position: the opaque pixel closest to the character's centroid."""
    alpha = target_np[0, 3]
    ys, xs = np.where(alpha > 0.5)
    if len(ys) == 0:
        return alpha.shape[0] // 2, alpha.shape[1] // 2
    centroid_y = ys.mean()
    centroid_x = xs.mean()
    dists = (ys - centroid_y) ** 2 + (xs - centroid_x) ** 2
    closest = dists.argmin()
    return int(ys[closest]), int(xs[closest])


def make_seed(channel_n, grid_size, seed_y=None, seed_x=None):
    """Seed state: all zeros, one pixel has alpha + hidden channels = 1.0."""
    seed = np.zeros([1, channel_n, grid_size, grid_size], np.float32)
    if seed_y is None:
        seed_y = grid_size // 2
    if seed_x is None:
        seed_x = grid_size // 2
    seed[0, 3:, seed_y, seed_x] = 1.0
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


def load_target(path, target_size, padding):
    """Load a target image, resize to target_size, pad to grid_size. Returns [1,4,H,W]."""
    img = Image.open(path).convert("RGBA")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    target = np.array(img, dtype=np.float32) / 255.0
    padded = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
    return padded.transpose(2, 0, 1)[np.newaxis]


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


def make_cond_vector(cond_id, n_targets, cond_dim, batch_size, device):
    """Create a one-hot condition vector for a single target id.

    Returns [batch_size, cond_dim] tensor with one-hot in the first n_targets dims.
    """
    cond = torch.zeros(batch_size, cond_dim, device=device)
    cond[:, cond_id] = 1.0
    return cond


def generate_test_target(char="R", output_path=None, target_size=40):
    """Generate a bold character on transparent background for testing."""
    img = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

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

    bbox = draw.textbbox((0, 0), char, font=font)
    tx = (target_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    ty = (target_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
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
        fg = target_np[:3, mask].mean(axis=1)
        return 1.0 - fg
    return np.array([0.0, 0.0, 0.0])


def save_visualization(x, target, step, vis_dir, batch_size=8):
    """Save batch states as a grid image."""
    rgba = to_rgba(x).detach().cpu().clamp(0, 1).numpy()
    target_np = target[0].cpu().numpy()
    bg = get_bg_color(target_np).reshape(1, 1, 3)

    n = min(len(rgba), batch_size)
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
      - weights.bin: raw concatenated float32 arrays

    When the model has cond_dim > 0, also exports the embedding table.
    """
    c1w = model.fc1.weight.detach().cpu().numpy()
    c1b = model.fc1.bias.detach().cpu().numpy()
    c2w = model.fc2.weight.detach().cpu().numpy()

    def to_b64(arr):
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

    weights_json = {
        "channel_n": model.channel_n,
        "hidden_n": model.hidden_n,
        "conv1_weight": {"shape": list(c1w.shape), "data_b64": to_b64(c1w)},
        "conv1_bias": {"shape": list(c1b.shape), "data_b64": to_b64(c1b)},
        "conv2_weight": {"shape": list(c2w.shape), "data_b64": to_b64(c2w)},
    }

    bin_parts = [c1w, c1b, c2w]

    if hasattr(model, "cond_dim") and model.cond_dim > 0:
        emb_w = model.cond_embedding.weight.detach().cpu().numpy()
        weights_json["cond_dim"] = model.cond_dim
        weights_json["num_classes"] = model.num_classes
        weights_json["cond_embedding"] = {
            "shape": list(emb_w.shape),
            "data_b64": to_b64(emb_w),
        }
        bin_parts.append(emb_w)

    json_path = os.path.join(output_dir, "weights.json")
    with open(json_path, "w") as f:
        json.dump(weights_json, f, indent=2)

    bin_path = os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        for arr in bin_parts:
            f.write(arr.astype(np.float32).tobytes())

    total_floats = sum(a.size for a in bin_parts)
    total_bytes = total_floats * 4
    print(f"Exported weights: {total_floats:,} floats ({total_bytes:,} bytes)")
    print(f"  {json_path}")
    print(f"  {bin_path}")


# --- Checkpointing ---


def save_checkpoint(path, step, model, optimizer, scheduler, pool_data, losses,
                    extra=None):
    """Save a training checkpoint."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "pool": pool_data,
        "losses": losses,
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load a training checkpoint.

    Returns:
        dict with keys: step, losses, pool, extra
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "step": ckpt["step"],
        "losses": ckpt.get("losses", []),
        "pool": ckpt.get("pool"),
        "extra": ckpt.get("extra"),
    }


# --- Loss Plot ---


def save_loss_plot(losses, output_dir):
    """Save a loss-over-time plot."""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=100)
    plt.close()
