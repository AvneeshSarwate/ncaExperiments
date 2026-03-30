"""Experiment 1: Conditional NCA — one shared model learns all target letters.

A single NCA model is conditioned on a per-letter embedding (broadcast every
step) so it can grow any of the target letters from the same center seed.

Usage:
    # Train with image files:
    uv run python -m experiments.train_exp1_conditional \
        --images images/A.png images/B.png images/C.png \
        --output-dir output/exp1_conditional \
        --steps 16000

    # Train with auto-generated characters:
    uv run python -m experiments.train_exp1_conditional \
        --chars A B C \
        --output-dir output/exp1_conditional \
        --steps 16000

    # Train from a YAML config:
    uv run python -m experiments.train_exp1_conditional \
        --config experiments/exp1_config.yaml

    # Resume from checkpoint:
    uv run python -m experiments.train_exp1_conditional \
        --images images/A.png images/B.png images/C.png \
        --output-dir output/exp1_conditional \
        --resume output/exp1_conditional/checkpoint_004000.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.common import (
    DEFAULTS,
    SamplePool,
    export_weights,
    generate_test_target,
    get_bg_color,
    get_device,
    load_target,
    loss_fn,
    make_circle_masks,
    make_seed,
    normalize_grads,
    to_rgba,
)
from experiments.nca_model import NCAModel
from experiments.rollouts import rollout_fixed

# Experiment-specific defaults (override base DEFAULTS)
EXP1_DEFAULTS = {
    **DEFAULTS,
    "steps": 16000,
    "cond_dim": 8,
}


# --- Visualization ---


def save_visualization(x_dict, targets, step, vis_dir, batch_size):
    """Save a grid showing samples from each letter side by side.

    Args:
        x_dict: dict mapping letter_name -> [B, C, H, W] tensor (detached, on CPU)
        targets: dict mapping letter_name -> [1, 4, H, W] numpy target
        step: current training step
        vis_dir: output directory for images
        batch_size: how many samples to show per letter
    """
    letter_names = sorted(x_dict.keys())
    n_letters = len(letter_names)
    n_show = min(batch_size, 4)
    cols = n_show + 1  # target + samples
    rows = n_letters

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(letter_names):
        target_np = targets[name][0]  # [4, H, W]
        bg = get_bg_color(target_np).reshape(1, 1, 3)

        # Target column
        ax = axes[row][0]
        rgb = target_np[:3].transpose(1, 2, 0)
        alpha = target_np[3:4].transpose(1, 2, 0)
        ax.imshow(rgb * alpha + bg * (1 - alpha), vmin=0, vmax=1)
        ax.set_title(f"Target {name}", fontsize=8)
        ax.axis("off")

        # Sample columns
        rgba = to_rgba(x_dict[name]).detach().cpu().clamp(0, 1).numpy()
        for j in range(n_show):
            ax = axes[row][j + 1]
            if j < len(rgba):
                img = rgba[j]
                rgb_s = img[:3].transpose(1, 2, 0)
                alpha_s = img[3:4].transpose(1, 2, 0)
                ax.imshow(rgb_s * alpha_s + bg * (1 - alpha_s), vmin=0, vmax=1)
                ax.set_title(f"Sample {j}", fontsize=8)
            ax.axis("off")

    fig.suptitle(f"Step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"step_{step:06d}.png"), dpi=100)
    plt.close()


# --- Training ---


def train(config: dict):
    """Train a conditional NCA model on multiple target letters.

    Args:
        config: dict with keys:
            images (list[str]): paths to RGBA target PNGs
            chars (list[str]|None): characters to auto-generate if no images
            output_dir (str): where to write outputs
            resume (str|None): checkpoint path to resume from
            + any keys from EXP1_DEFAULTS to override
    """
    p = {**EXP1_DEFAULTS, **config}
    output_dir = Path(p["output_dir"])
    resume = p.get("resume")
    grid_size = p["target_size"] + 2 * p["target_padding"]

    device = get_device()
    print(f"Device: {device}")

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --- Build letter list + targets ---
    image_paths = p.get("images") or []
    chars = p.get("chars") or []

    if not image_paths and not chars:
        raise ValueError("Provide --images or --chars (or both via --config)")

    letter_names = []
    targets_np = {}  # name -> [1, 4, H, W] numpy

    for img_path in image_paths:
        name = Path(img_path).stem
        targets_np[name] = load_target(img_path, p["target_size"], p["target_padding"])
        letter_names.append(name)
        print(f"Loaded target: {img_path} -> '{name}'")

    for ch in chars:
        if ch in targets_np:
            continue
        gen_path = output_dir / f"target_{ch}.png"
        generate_test_target(ch, str(gen_path), p["target_size"])
        targets_np[ch] = load_target(str(gen_path), p["target_size"], p["target_padding"])
        letter_names.append(ch)
        print(f"Generated test target for '{ch}' at {gen_path}")

    num_classes = len(letter_names)
    print(f"Letters ({num_classes}): {letter_names}")

    # Pre-compute target tensors on device
    targets_t = {}
    for name in letter_names:
        targets_t[name] = torch.from_numpy(targets_np[name]).to(device)

    # --- Model ---
    cond_dim = p["cond_dim"]
    model = NCAModel(
        channel_n=p["channel_n"],
        hidden_n=p["hidden_n"],
        fire_rate=p["fire_rate"],
        cond_dim=cond_dim,
        num_classes=num_classes,
    ).to(device)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # --- Optimizer + LR schedule ---
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr_initial"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 if step < p["lr_decay_step"] else p["lr_decay_factor"],
    )

    # --- Fixed center seed (shared across all letters) ---
    center = grid_size // 2
    seed = make_seed(p["channel_n"], grid_size, center, center)
    print(f"Seed position: ({center}, {center}) — fixed center for all letters")

    # --- Per-letter sample pools ---
    pools = {}
    for name in letter_names:
        pools[name] = SamplePool(x=np.repeat(seed, p["pool_size"], axis=0))

    # Letter name -> index mapping
    letter_to_idx = {name: i for i, name in enumerate(letter_names)}

    # --- Resume ---
    start_step = 0
    losses = []
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if "pools" in ckpt:
            for name in letter_names:
                if name in ckpt["pools"]:
                    pools[name] = SamplePool(x=ckpt["pools"][name])
        if "losses" in ckpt:
            losses = ckpt["losses"]
        print(f"Resumed from step {start_step}")

    # --- Save config for reproducibility ---
    config_dump = {
        "letter_names": letter_names,
        "num_classes": num_classes,
        "cond_dim": cond_dim,
        "grid_size": grid_size,
        "seed_position": [center, center],
    }
    for k in DEFAULTS:
        config_dump[k] = p[k]
    config_dump["steps"] = p["steps"]
    with open(str(output_dir / "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    # --- Training loop ---
    t0 = time.time()
    model.train()
    total_steps = p["steps"]
    batch_size = p["batch_size"]

    # Keep track of last rollout per letter for visualization
    last_x = {}

    prev_ckpt_path = None
    for step in range(start_step, total_steps):
        # 1. Sample a letter uniformly at random
        letter_idx_int = np.random.randint(num_classes)
        name = letter_names[letter_idx_int]
        target_batch = targets_t[name].expand(batch_size, -1, -1, -1)

        # 2. Sample a batch from that letter's pool
        batch = pools[name].sample(batch_size)
        x0 = batch.x.copy()

        # 3. Sort by loss (highest first), replace worst with seed, apply damage
        with torch.no_grad():
            x0_t = torch.from_numpy(x0).to(device)
            sample_losses = loss_fn(x0_t, target_batch).cpu().numpy()
        loss_rank = sample_losses.argsort()[::-1].copy()
        x0 = x0[loss_rank]

        x0[:1] = seed

        if p["damage_n"] > 0:
            masks = make_circle_masks(p["damage_n"], grid_size, grid_size)
            damage = 1.0 - masks[:, np.newaxis, :, :]
            x0[-p["damage_n"]:] *= damage

        # 4. Rollout
        x = torch.from_numpy(x0).to(device)
        iter_n = np.random.randint(p["min_nca_steps"], p["max_nca_steps"])
        x = rollout_fixed(model, x, iter_n, cond_ids=letter_idx_int)

        # 5. Loss = MSE on RGBA vs that letter's target
        loss_per_sample = loss_fn(x, target_batch)
        loss = loss_per_sample.mean()

        # 6. Backprop with per-variable gradient normalization
        optimizer.zero_grad()
        loss.backward()
        normalize_grads(model)
        optimizer.step()
        scheduler.step()

        # 7. Commit results back to the letter's pool
        batch.x[loss_rank] = x.detach().cpu().numpy()
        batch.commit()

        loss_val = loss.item()
        losses.append(loss_val)

        # Store latest rollout for visualization
        last_x[name] = x.detach()

        if step % 100 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step:5d}/{total_steps}"
                f"  letter={name}"
                f"  loss={loss_val:.6f}"
                f"  lr={lr:.1e}"
                f"  ({elapsed:.0f}s)"
            )

        if step % 500 == 0:
            # Generate visualization samples from all letters
            vis_x = {}
            model.eval()
            with torch.no_grad():
                for li, ln in enumerate(letter_names):
                    vis_seed = torch.from_numpy(np.repeat(seed, batch_size, axis=0)).to(device)
                    vis_out = rollout_fixed(model, vis_seed, p["max_nca_steps"], cond_ids=li)
                    vis_x[ln] = vis_out
            model.train()
            save_visualization(vis_x, targets_np, step, str(vis_dir), batch_size)

        if step > 0 and step % 2000 == 0:
            if prev_ckpt_path and os.path.exists(prev_ckpt_path):
                os.remove(prev_ckpt_path)
                prev_ckpt_path = None
            ckpt_path = str(output_dir / f"checkpoint_{step:06d}.pt")
            pool_data = {name: pools[name].x for name in letter_names}
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "pools": pool_data,
                    "losses": losses,
                    "letter_names": letter_names,
                },
                ckpt_path,
            )
            prev_ckpt_path = ckpt_path
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Final outputs ---
    if prev_ckpt_path and os.path.exists(prev_ckpt_path):
        os.remove(prev_ckpt_path)

    total_time = time.time() - t0
    print(f"\nTraining complete. {total_steps} steps in {total_time:.0f}s")

    # Final checkpoint
    torch.save(
        {
            "step": total_steps,
            "model": model.state_dict(),
            "losses": losses,
            "letter_names": letter_names,
        },
        str(output_dir / "checkpoint_final.pt"),
    )

    # Export weights (including embedding — cond_dim/num_classes auto-detected)
    export_weights(model, str(output_dir))

    # Loss curve
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
    vis_x = {}
    model.eval()
    with torch.no_grad():
        for li, ln in enumerate(letter_names):
            vis_seed = torch.from_numpy(np.repeat(seed, batch_size, axis=0)).to(device)
            vis_out = rollout_fixed(model, vis_seed, p["max_nca_steps"], cond_ids=li)
            vis_x[ln] = vis_out
    save_visualization(vis_x, targets_np, total_steps, str(vis_dir), batch_size)

    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Conditional NCA — one model for all letters"
    )
    parser.add_argument(
        "--images", type=str, nargs="+",
        help="Paths to RGBA target PNGs (letter name inferred from filename stem)",
    )
    parser.add_argument(
        "--chars", type=str, nargs="+",
        help="Characters to auto-generate as targets (alternative to --images)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/exp1_conditional",
        help="Output directory (default: output/exp1_conditional)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Training steps (default: 16000)",
    )
    parser.add_argument(
        "--cond-dim", type=int, default=None,
        help="Conditioning embedding dimension (default: 8)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (CLI args override YAML values)",
    )
    args = parser.parse_args()

    # Build config: YAML base -> CLI overrides
    config = {}
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        if "params" in yaml_cfg:
            config.update(yaml_cfg["params"])
        if "images" in yaml_cfg:
            config["images"] = yaml_cfg["images"]
        if "chars" in yaml_cfg:
            config["chars"] = yaml_cfg["chars"]
        if "output_dir" in yaml_cfg:
            config["output_dir"] = yaml_cfg["output_dir"]

    # CLI overrides
    if args.images:
        config["images"] = args.images
    if args.chars:
        config["chars"] = args.chars
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.steps is not None:
        config["steps"] = args.steps
    if args.cond_dim is not None:
        config["cond_dim"] = args.cond_dim
    if args.resume:
        config["resume"] = args.resume

    # Ensure output_dir has a default
    config.setdefault("output_dir", "output/exp1_conditional")

    train(config)
