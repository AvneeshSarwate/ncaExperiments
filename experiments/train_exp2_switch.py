"""Training script for Experiment 2: Switch Curriculum.

Builds on the conditional NCA from Exp1 and adds A→B target switching
during training to learn morphing.

Two-phase training:
  Phase 1 (warmup): Pure Exp1-style fixed-target training.
                     Skipped if --pretrained is provided.
  Phase 2 (switch):  Each step has switch_ratio probability of doing a
                     switch batch (A→B transition), otherwise a normal
                     fixed-target batch.

Usage:
    # Train from scratch:
    uv run python -m experiments.train_exp2_switch \
        --images images/A.png images/B.png images/C.png \
        --output-dir output/exp2_switch --steps 24000

    # Start from Exp1 pretrained checkpoint:
    uv run python -m experiments.train_exp2_switch \
        --images images/A.png images/B.png images/C.png \
        --output-dir output/exp2_switch --steps 24000 \
        --pretrained output/exp1_conditional/checkpoint_final.pt

    # Resume interrupted Exp2 training:
    uv run python -m experiments.train_exp2_switch \
        --output-dir output/exp2_switch \
        --resume output/exp2_switch/checkpoint_012000.pt
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
    grid_size,
    load_target,
    loss_fn,
    make_circle_masks,
    make_seed,
    normalize_grads,
    to_rgba,
)
from experiments.nca_model import NCAModel
from experiments.rollouts import rollout_fixed, rollout_switch

# --- Exp2 defaults ---
EXP2_DEFAULTS = {
    **DEFAULTS,
    "cond_dim": 8,
    "steps": 24000,
    "warmup_steps": 4000,
    "switch_ratio": 0.75,
    "steps_a_min": 32,
    "steps_a_max": 48,
    "steps_b_min": 48,
    "steps_b_max": 64,
}


# --- Visualization ---


def save_vis_grid(states_rgba, names, step, vis_dir):
    """Save a grid of RGBA images with labels."""
    n = len(states_rgba)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            img = states_rgba[idx]
            rgb = img[:3].transpose(1, 2, 0)
            alpha = img[3:4].transpose(1, 2, 0)
            bg = np.array([0.0, 0.0, 0.0]).reshape(1, 1, 3)
            ax.imshow(rgb * alpha + bg * (1 - alpha), vmin=0, vmax=1)
            ax.set_title(names[idx], fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"step_{step:06d}.png"), dpi=100)
    plt.close()


def save_switch_visualization(
    model, seeds, targets_np, target_names, n_targets, step, vis_dir, device
):
    """Show switch results: for a few (A,B) pairs, show A-phase and B-phase side by side."""
    model.eval()
    pairs = []
    for a in range(min(n_targets, 3)):
        for b in range(min(n_targets, 3)):
            if a != b:
                pairs.append((a, b))
    if len(pairs) > 6:
        pairs = pairs[:6]

    states_rgba = []
    names = []

    with torch.no_grad():
        for a_id, b_id in pairs:
            x = torch.from_numpy(seeds[a_id]).to(device)
            cond_a = torch.tensor([a_id], device=device)
            cond_b = torch.tensor([b_id], device=device)

            # Phase A
            x_a = rollout_fixed(model, x, 40, cond_ids=cond_a)
            states_rgba.append(to_rgba(x_a).cpu().clamp(0, 1).numpy()[0])
            names.append(f"{target_names[a_id]}→(A phase)")

            # Reset hidden + Phase B
            x_reset = NCAModel.reset_hidden(x_a)
            x_b = rollout_fixed(model, x_reset, 56, cond_ids=cond_b)
            states_rgba.append(to_rgba(x_b).cpu().clamp(0, 1).numpy()[0])
            names.append(f"{target_names[a_id]}→{target_names[b_id]}")

    save_vis_grid(states_rgba, names, step, vis_dir)
    model.train()


def save_fixed_visualization(x, targets_np, target_idx, target_names, step, vis_dir):
    """Save batch states for a fixed-target step."""
    rgba = to_rgba(x).detach().cpu().clamp(0, 1).numpy()
    n = min(len(rgba), 8)

    states = []
    names = []

    # Show target
    t_np = targets_np[target_idx][0]
    states.append(t_np)
    names.append(f"Target: {target_names[target_idx]}")

    for i in range(min(n, 5)):
        states.append(rgba[i])
        names.append(f"Sample {i}")

    save_vis_grid(states, names, step, vis_dir)


# --- Training ---


def train(config: dict):
    p = {**EXP2_DEFAULTS, **config}
    output_dir = Path(p["output_dir"])
    resume = p.get("resume")
    pretrained = p.get("pretrained")
    image_paths = p.get("images", [])
    chars = p.get("chars", [])

    device = get_device()
    print(f"Device: {device}")

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load targets
    targets_np = []
    target_names = []

    if not image_paths and not chars and resume:
        # Will be restored from checkpoint config
        pass
    elif not image_paths and chars:
        for ch in chars:
            gen_path = output_dir / f"target_{ch}.png"
            generate_test_target(ch, str(gen_path), p["target_size"])
            targets_np.append(load_target(str(gen_path), p["target_size"], p["target_padding"]))
            target_names.append(ch)
            print(f"Generated test target for '{ch}' at {gen_path}")
    else:
        for img_path in image_paths:
            targets_np.append(load_target(img_path, p["target_size"], p["target_padding"]))
            target_names.append(Path(img_path).stem)
            print(f"Loaded target: {img_path}")

    n_targets = len(targets_np)
    if n_targets < 2 and not resume:
        raise ValueError("Need at least 2 targets for switch training")

    gs = grid_size(p["target_size"], p["target_padding"])
    cond_dim = p["cond_dim"]

    # Targets as tensors
    targets_t = [torch.from_numpy(t).to(device) for t in targets_np]

    # Fixed center seed (shared across all letters for morphing compatibility)
    center = gs // 2
    seed = make_seed(p["channel_n"], gs, center, center)
    seeds = [seed] * n_targets
    print(f"  Seed position: ({center}, {center}) — fixed center for all letters")

    # Per-target pools
    pools = [
        SamplePool(x=np.repeat(seeds[i], p["pool_size"], axis=0))
        for i in range(n_targets)
    ]

    # Model
    model = NCAModel(
        p["channel_n"], p["hidden_n"], p["fire_rate"],
        cond_dim=cond_dim, num_classes=n_targets,
    ).to(device)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer + LR schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr_initial"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 if step < p["lr_decay_step"] else p["lr_decay_factor"],
    )

    # Load pretrained Exp1 checkpoint (skip warmup)
    warmup_steps = p["warmup_steps"]
    if pretrained and not resume:
        ckpt = torch.load(pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded pretrained weights from {pretrained}")
        warmup_steps = 0  # Skip warmup when starting from pretrained
        # Re-create optimizer with the loaded model params
        optimizer = torch.optim.Adam(model.parameters(), lr=p["lr_initial"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 if step < p["lr_decay_step"] else p["lr_decay_factor"],
        )

    # Resume
    start_step = 0
    losses = []
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        if "losses" in ckpt:
            losses = ckpt["losses"]
        if "pools" in ckpt:
            for i, pool_data in enumerate(ckpt["pools"]):
                pools[i] = SamplePool(x=pool_data)
        if "warmup_steps" in ckpt:
            warmup_steps = ckpt["warmup_steps"]
        if "target_names" in ckpt and not target_names:
            target_names = ckpt["target_names"]
        if "n_targets" in ckpt and not targets_np:
            n_targets = ckpt["n_targets"]
        print(f"Resumed from step {start_step}")

    # Save config
    config_out = {
        "n_targets": n_targets,
        "target_names": target_names,
        "cond_dim": cond_dim,
        "warmup_steps": warmup_steps,
        "switch_ratio": p["switch_ratio"],
        "steps_a_range": [p["steps_a_min"], p["steps_a_max"]],
        "steps_b_range": [p["steps_b_min"], p["steps_b_max"]],
        "total_steps": p["steps"],
        "channel_n": p["channel_n"],
        "hidden_n": p["hidden_n"],
        "batch_size": p["batch_size"],
        "pool_size": p["pool_size"],
    }
    with open(str(output_dir / "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    # Training loop
    t0 = time.time()
    model.train()
    total_steps = p["steps"]
    batch_size = p["batch_size"]

    prev_ckpt_path = None
    for step in range(start_step, total_steps):
        in_warmup = step < warmup_steps

        if in_warmup or np.random.random() > p["switch_ratio"]:
            # --- Fixed-target batch (Exp1-style) ---
            target_idx = np.random.randint(n_targets)
            pool = pools[target_idx]
            batch = pool.sample(batch_size)
            x0 = batch.x.copy()

            target_batch = targets_t[target_idx].expand(batch_size, -1, -1, -1)

            # Sort by loss, replace worst with seed
            with torch.no_grad():
                x0_t = torch.from_numpy(x0).to(device)
                sample_losses = loss_fn(x0_t, target_batch).cpu().numpy()
            loss_rank = sample_losses.argsort()[::-1].copy()
            x0 = x0[loss_rank]
            x0[:1] = seeds[target_idx]

            # Damage
            if p["damage_n"] > 0:
                masks = make_circle_masks(p["damage_n"], gs, gs)
                damage = 1.0 - masks[:, np.newaxis, :, :]
                x0[-p["damage_n"]:] *= damage

            x = torch.from_numpy(x0).to(device)
            cond_ids = torch.full((batch_size,), target_idx, dtype=torch.long, device=device)
            iter_n = np.random.randint(p["min_nca_steps"], p["max_nca_steps"])

            x = rollout_fixed(model, x, iter_n, cond_ids=cond_ids)

            loss_per_sample = loss_fn(x, target_batch)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            normalize_grads(model)
            optimizer.step()
            scheduler.step()

            batch.x[loss_rank] = x.detach().cpu().numpy()
            batch.commit()

        else:
            # --- Switch batch (A→B) ---
            a_id, b_id = np.random.choice(n_targets, 2, replace=False)

            pool_a = pools[a_id]
            batch = pool_a.sample(batch_size)
            x0 = batch.x.copy()

            target_b_batch = targets_t[b_id].expand(batch_size, -1, -1, -1)

            # Sort by loss vs target A, replace worst with seed
            target_a_batch = targets_t[a_id].expand(batch_size, -1, -1, -1)
            with torch.no_grad():
                x0_t = torch.from_numpy(x0).to(device)
                sample_losses = loss_fn(x0_t, target_a_batch).cpu().numpy()
            loss_rank = sample_losses.argsort()[::-1].copy()
            x0 = x0[loss_rank]
            x0[:1] = seeds[a_id]

            # Damage
            if p["damage_n"] > 0:
                masks = make_circle_masks(p["damage_n"], gs, gs)
                damage = 1.0 - masks[:, np.newaxis, :, :]
                x0[-p["damage_n"]:] *= damage

            x = torch.from_numpy(x0).to(device)
            cond_a = torch.full((batch_size,), a_id, dtype=torch.long, device=device)
            cond_b = torch.full((batch_size,), b_id, dtype=torch.long, device=device)

            steps_a = np.random.randint(p["steps_a_min"], p["steps_a_max"])
            steps_b = np.random.randint(p["steps_b_min"], p["steps_b_max"])

            x = rollout_switch(model, x, steps_a, cond_a, steps_b, cond_b)

            loss_per_sample = loss_fn(x, target_b_batch)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            normalize_grads(model)
            optimizer.step()
            scheduler.step()

            # Commit final states to B's pool
            pools[b_id].x[np.random.choice(p["pool_size"], batch_size, replace=False)] = (
                x.detach().cpu().numpy()
            )

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            phase = "warmup" if in_warmup else "switch"
            print(
                f"  Step {step:5d}/{total_steps}"
                f"  loss={loss_val:.6f}"
                f"  lr={lr:.1e}"
                f"  phase={phase}"
                f"  ({elapsed:.0f}s)"
            )

        if step % 500 == 0:
            if step >= warmup_steps:
                save_switch_visualization(
                    model, seeds, targets_np, target_names,
                    n_targets, step, str(vis_dir), device,
                )
            else:
                # During warmup, show a random fixed-target vis
                target_idx = np.random.randint(n_targets)
                pool = pools[target_idx]
                with torch.no_grad():
                    vis_batch = pool.sample(min(batch_size, 4))
                    vis_x = torch.from_numpy(vis_batch.x).to(device)
                save_fixed_visualization(
                    vis_x, targets_np, target_idx, target_names, step, str(vis_dir),
                )

        if step > 0 and step % 2000 == 0:
            if prev_ckpt_path and os.path.exists(prev_ckpt_path):
                os.remove(prev_ckpt_path)
                prev_ckpt_path = None
            ckpt_path = str(output_dir / f"checkpoint_{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "pools": [pool.x for pool in pools],
                    "losses": losses,
                    "warmup_steps": warmup_steps,
                    "switch_ratio": p["switch_ratio"],
                    "phase": "warmup" if step < warmup_steps else "switch",
                    "n_targets": n_targets,
                    "target_names": target_names,
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

    torch.save(
        {
            "step": total_steps,
            "model": model.state_dict(),
            "losses": losses,
            "warmup_steps": warmup_steps,
            "switch_ratio": p["switch_ratio"],
            "phase": "switch",
            "n_targets": n_targets,
            "target_names": target_names,
        },
        str(output_dir / "checkpoint_final.pt"),
    )

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

    # Final switch visualization
    save_switch_visualization(
        model, seeds, targets_np, target_names,
        n_targets, total_steps, str(vis_dir), device,
    )

    print(f"Outputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NCA Exp2: Switch Curriculum for morphing"
    )
    parser.add_argument(
        "--images", type=str, nargs="+", help="Paths to RGBA target PNGs"
    )
    parser.add_argument(
        "--chars", type=str, nargs="+",
        help="Characters to generate as test targets (if --images not given)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/exp2_switch",
        help="Output directory",
    )
    parser.add_argument(
        "--steps", type=int, default=EXP2_DEFAULTS["steps"],
        help="Total training steps (default: 24000)",
    )
    parser.add_argument(
        "--pretrained", type=str,
        help="Path to Exp1 pretrained checkpoint (skips warmup)",
    )
    parser.add_argument(
        "--resume", type=str, help="Path to Exp2 checkpoint to resume from",
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML config file",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=EXP2_DEFAULTS["warmup_steps"],
        help="Warmup steps (fixed-target only, default: 4000)",
    )
    parser.add_argument(
        "--switch-ratio", type=float, default=EXP2_DEFAULTS["switch_ratio"],
        help="Probability of switch batch in phase 2 (default: 0.75)",
    )
    parser.add_argument(
        "--steps-a-min", type=int, default=EXP2_DEFAULTS["steps_a_min"],
        help="Min steps under cond A (default: 32)",
    )
    parser.add_argument(
        "--steps-a-max", type=int, default=EXP2_DEFAULTS["steps_a_max"],
        help="Max steps under cond A (default: 48)",
    )
    parser.add_argument(
        "--steps-b-min", type=int, default=EXP2_DEFAULTS["steps_b_min"],
        help="Min steps under cond B (default: 48)",
    )
    parser.add_argument(
        "--steps-b-max", type=int, default=EXP2_DEFAULTS["steps_b_max"],
        help="Max steps under cond B (default: 64)",
    )
    args = parser.parse_args()

    # Build config from CLI + optional YAML
    cfg = {}
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        if "params" in yaml_cfg:
            cfg.update(yaml_cfg["params"])
        if "images" in yaml_cfg and not args.images:
            args.images = yaml_cfg["images"]

    # CLI overrides
    cfg["output_dir"] = args.output_dir
    cfg["steps"] = args.steps
    cfg["resume"] = args.resume
    cfg["pretrained"] = args.pretrained
    cfg["warmup_steps"] = args.warmup_steps
    cfg["switch_ratio"] = args.switch_ratio
    cfg["steps_a_min"] = args.steps_a_min
    cfg["steps_a_max"] = args.steps_a_max
    cfg["steps_b_min"] = args.steps_b_min
    cfg["steps_b_max"] = args.steps_b_max

    if args.images:
        cfg["images"] = args.images
    if args.chars:
        cfg["chars"] = args.chars

    train(cfg)
