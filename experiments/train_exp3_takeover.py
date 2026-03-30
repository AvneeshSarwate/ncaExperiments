"""Experiment 3: Foreign-State Takeover.

Per-letter NCA models trained to recover from arbitrary garbage states,
including other letters' visible RGBA output. Uses a two-stage approach:

Stage 1 — Generate donor states:
    Load baseline per-letter checkpoints from --donors-dir and run each
    model forward from seed for random steps, collecting RGBA snapshots
    into a donor bank.

Stage 2 — Takeover fine-tuning:
    For each target letter, train a separate NCAModel (no conditioning).
    Each batch mixes three state types:
      1. Fresh seed (standard seed state)
      2. Same-letter pool (normal pool-based training)
      3. Foreign RGBA (donor letter snapshot → channels 0:4, zero 4:16)

Usage:
    uv run python -m experiments.train_exp3_takeover \
        --images images/A.png images/B.png images/C.png \
        --donors-dir output/ \
        --output-dir output/exp3_takeover \
        --steps 10000

    uv run python -m experiments.train_exp3_takeover \
        --config train_config.yaml \
        --donors-dir output/ \
        --output-dir output/exp3_takeover

    uv run python -m experiments.train_exp3_takeover \
        --chars A B C \
        --donors-dir output/ \
        --output-dir output/exp3_takeover
"""

import argparse
import json
import os
import sys
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
    find_seed_position,
    get_device,
    load_target,
    loss_fn,
    make_circle_masks,
    make_seed,
    normalize_grads,
    save_visualization,
)
from experiments.nca_model import NCAModel

# --- Donor Bank Generation ---


def generate_donor_bank(
    checkpoint_path,
    target_path,
    bank_size=256,
    channel_n=16,
    hidden_n=128,
    fire_rate=0.5,
    target_size=40,
    target_padding=16,
    min_steps=64,
    max_steps=96,
    device=None,
):
    """Generate a bank of RGBA snapshots by running a trained model forward.

    Returns:
        np.ndarray of shape [bank_size, 4, H, W] (float32 RGBA snapshots)
    """
    if device is None:
        device = get_device()

    grid_size = target_size + 2 * target_padding

    # Load trained model
    model = NCAModel(channel_n, hidden_n, fire_rate).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load target to find seed position
    target_np = load_target(target_path, target_size, target_padding)
    seed_y, seed_x = find_seed_position(target_np, target_padding)
    seed = make_seed(channel_n, grid_size, seed_y, seed_x)

    bank = np.zeros([bank_size, 4, grid_size, grid_size], dtype=np.float32)

    with torch.no_grad():
        # Process in batches of 32 for efficiency
        batch_sz = min(32, bank_size)
        for start in range(0, bank_size, batch_sz):
            end = min(start + batch_sz, bank_size)
            n = end - start

            x = torch.from_numpy(np.repeat(seed, n, axis=0)).to(device)
            steps = np.random.randint(min_steps, max_steps + 1)
            for _ in range(steps):
                x = model(x)

            bank[start:end] = x[:, :4].cpu().clamp(0, 1).numpy()

    return bank


def load_donor_banks(
    donors_dir, image_paths, bank_size=256, params=None, device=None
):
    """Load or generate donor banks for all letters.

    Args:
        donors_dir: directory with per-letter baseline outputs (e.g. output/)
        image_paths: list of target image paths
        bank_size: number of RGBA snapshots per donor letter
        params: hyperparameters dict
        device: torch device

    Returns:
        dict mapping letter name -> np.ndarray [bank_size, 4, H, W]
    """
    if params is None:
        params = DEFAULTS
    if device is None:
        device = get_device()

    banks = {}
    for img_path in image_paths:
        name = Path(img_path).stem
        donor_dir = Path(donors_dir) / name
        ckpt_path = donor_dir / "checkpoint_final.pt"

        if not ckpt_path.exists():
            print(f"  WARNING: No checkpoint for donor '{name}' at {ckpt_path}, skipping")
            continue

        print(f"  Generating donor bank for '{name}' ({bank_size} snapshots)...")
        bank = generate_donor_bank(
            str(ckpt_path),
            img_path,
            bank_size=bank_size,
            channel_n=params["channel_n"],
            hidden_n=params["hidden_n"],
            fire_rate=params["fire_rate"],
            target_size=params["target_size"],
            target_padding=params["target_padding"],
            min_steps=params["min_nca_steps"],
            max_steps=params["max_nca_steps"],
            device=device,
        )
        banks[name] = bank
        print(f"    -> {bank.shape}")

    return banks


# --- Batch Composition ---


def compose_batch(
    seed,
    pool,
    donor_banks,
    target_letter,
    batch_size,
    channel_n,
    grid_size,
    mix_seed=0.33,
    mix_same=0.34,
    mix_foreign=0.33,
):
    """Compose a training batch from three state types.

    Returns:
        x0: np.ndarray [batch_size, channel_n, H, W]
        batch: SamplePool sub-batch (for commit)
    """
    n_seed = max(1, int(round(batch_size * mix_seed)))
    n_same = max(1, int(round(batch_size * mix_same)))
    n_foreign = batch_size - n_seed - n_same

    # Sample from pool for the same-letter portion
    batch = pool.sample(batch_size)
    x0 = batch.x.copy()

    # 1) Fresh seed slots
    x0[:n_seed] = seed

    # 2) Same-letter pool slots — already filled from pool sample (indices n_seed : n_seed+n_same)
    # (no change needed, these come from the pool)

    # 3) Foreign RGBA slots
    if n_foreign > 0:
        # Get donor letters (excluding target)
        foreign_names = [k for k in donor_banks if k != target_letter]
        if foreign_names:
            for i in range(n_foreign):
                slot_idx = n_seed + n_same + i
                # Pick a random donor letter
                donor_name = foreign_names[np.random.randint(len(foreign_names))]
                donor_bank = donor_banks[donor_name]
                # Pick a random snapshot
                snap_idx = np.random.randint(len(donor_bank))
                rgba_snap = donor_bank[snap_idx]  # [4, H, W]
                # Set channels 0:4 to donor RGBA, zero channels 4:16
                x0[slot_idx] = 0.0
                x0[slot_idx, :4] = rgba_snap
        else:
            # No foreign donors available — fill with seed instead
            x0[n_seed + n_same :] = seed

    return x0, batch


# --- Per-Letter Training ---


def train_letter(
    target_path,
    output_dir,
    donor_banks,
    params,
    resume=None,
    device=None,
):
    """Train a single letter's takeover model.

    Args:
        target_path: path to target RGBA PNG
        output_dir: per-letter output directory
        donor_banks: dict of donor banks (letter name -> [N, 4, H, W])
        params: merged hyperparameters dict
        resume: optional checkpoint path to resume from
        device: torch device
    """
    if device is None:
        device = get_device()

    letter_name = Path(target_path).stem
    output_dir = Path(output_dir)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    grid_size = params["target_size"] + 2 * params["target_padding"]

    # Load target
    target_np = load_target(target_path, params["target_size"], params["target_padding"])
    target = torch.from_numpy(target_np).to(device)
    target_batch = target.expand(params["batch_size"], -1, -1, -1)
    print(f"  Target: {target_path}")

    # Model (no conditioning — standard NCAModel)
    model = NCAModel(
        params["channel_n"], params["hidden_n"], params["fire_rate"]
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Optimizer + LR schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr_initial"])
    decay_step = params["lr_decay_step"]
    decay_factor = params["lr_decay_factor"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 if step < decay_step else decay_factor,
    )

    # Seed and pool
    seed_y, seed_x = find_seed_position(target_np, params["target_padding"])
    print(f"  Seed position: ({seed_x}, {seed_y})")
    seed = make_seed(params["channel_n"], grid_size, seed_y, seed_x)
    pool = SamplePool(x=np.repeat(seed, params["pool_size"], axis=0))

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
        if "pool" in ckpt:
            pool = SamplePool(x=ckpt["pool"])
        if "losses" in ckpt:
            losses = ckpt["losses"]
        print(f"  Resumed from step {start_step}")

    # Training loop
    t0 = time.time()
    model.train()
    total_steps = params["steps"]

    mix_seed = params.get("mix_seed", 0.33)
    mix_same = params.get("mix_same", 0.34)
    mix_foreign = params.get("mix_foreign", 0.33)

    prev_ckpt_path = None
    x = None
    for step in range(start_step, total_steps):
        # Compose batch with three state types
        x0, batch = compose_batch(
            seed,
            pool,
            donor_banks,
            letter_name,
            params["batch_size"],
            params["channel_n"],
            grid_size,
            mix_seed=mix_seed,
            mix_same=mix_same,
            mix_foreign=mix_foreign,
        )

        # Damage some samples (applied to pool-sourced slots)
        if params["damage_n"] > 0:
            n_seed_slots = max(1, int(round(params["batch_size"] * mix_seed)))
            masks = make_circle_masks(params["damage_n"], grid_size, grid_size)
            damage = 1.0 - masks[:, np.newaxis, :, :]
            x0[-params["damage_n"] :] *= damage

        x = torch.from_numpy(x0).to(device)
        iter_n = np.random.randint(params["min_nca_steps"], params["max_nca_steps"])

        for _ in range(iter_n):
            x = model(x)

        loss_per_sample = loss_fn(x, target_batch)
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        normalize_grads(model)
        optimizer.step()
        scheduler.step()

        # Commit results back to pool
        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 100 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step:5d}/{total_steps}"
                f"  loss={loss_val:.6f}"
                f"  lr={lr:.1e}"
                f"  ({elapsed:.0f}s)"
            )

        if step % 500 == 0:
            save_visualization(x, target, step, str(vis_dir))

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
                    "pool": pool.x,
                    "losses": losses,
                },
                ckpt_path,
            )
            prev_ckpt_path = ckpt_path
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Final outputs ---
    if prev_ckpt_path and os.path.exists(prev_ckpt_path):
        os.remove(prev_ckpt_path)

    total_time = time.time() - t0
    print(f"  Training complete. {total_steps} steps in {total_time:.0f}s")

    torch.save(
        {"step": total_steps, "model": model.state_dict(), "losses": losses},
        str(output_dir / "checkpoint_final.pt"),
    )

    export_weights(model, str(output_dir))

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "loss.png"), dpi=100)
    plt.close()

    if x is not None:
        save_visualization(x, target, total_steps, str(vis_dir))

    print(f"  Outputs saved to {output_dir}/")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Foreign-State Takeover — train per-letter NCA models "
        "to recover from arbitrary garbage states"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Target image paths (e.g. images/A.png images/B.png)",
    )
    parser.add_argument(
        "--chars",
        nargs="+",
        help="Characters to train (uses images/<CHAR>.png)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML config file (same format as train_config.yaml)",
    )
    parser.add_argument(
        "--donors-dir",
        type=str,
        required=True,
        help="Directory with per-letter baseline outputs (each subfolder has checkpoint_final.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/exp3_takeover",
        help="Output directory (default: output/exp3_takeover)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training steps per letter (overrides config/default)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume letter from checkpoint (only valid for single-letter training)",
    )
    parser.add_argument(
        "--donor-bank-size",
        type=int,
        default=256,
        help="Number of RGBA snapshots to pre-generate per donor letter (default: 256)",
    )
    parser.add_argument(
        "--mix-seed",
        type=float,
        default=0.33,
        help="Fraction of batch that's fresh seed (default: 0.33)",
    )
    parser.add_argument(
        "--mix-same",
        type=float,
        default=0.34,
        help="Fraction of batch that's same-letter pool (default: 0.34)",
    )
    parser.add_argument(
        "--mix-foreign",
        type=float,
        default=0.33,
        help="Fraction of batch that's foreign RGBA (default: 0.33)",
    )
    args = parser.parse_args()

    # Resolve image paths
    image_paths = []
    params = dict(DEFAULTS)

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        params.update(cfg.get("params") or {})
        if cfg.get("images"):
            image_paths = cfg["images"]

    if args.images:
        image_paths = args.images

    if args.chars:
        image_paths = [f"images/{c}.png" for c in args.chars]

    if not image_paths:
        print("ERROR: No images specified. Use --images, --chars, or --config.")
        sys.exit(1)

    # Verify all image paths exist
    for img in image_paths:
        if not Path(img).exists():
            print(f"ERROR: Image not found: {img}")
            sys.exit(1)

    # Override params from CLI
    if args.steps is not None:
        params["steps"] = args.steps
    params["mix_seed"] = args.mix_seed
    params["mix_same"] = args.mix_same
    params["mix_foreign"] = args.mix_foreign

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"=== Experiment 3: Foreign-State Takeover ===")
    print(f"Device: {device}")
    print(f"Letters: {', '.join(Path(p).stem for p in image_paths)}")
    print(f"Donors dir: {args.donors_dir}")
    print(f"Donor bank size: {args.donor_bank_size}")
    print(f"Mix: seed={args.mix_seed:.2f}, same={args.mix_same:.2f}, foreign={args.mix_foreign:.2f}")
    print(f"Steps per letter: {params['steps']}")
    print(f"Output: {output_dir}/")
    print()

    # Stage 1: Generate donor banks
    print("--- Stage 1: Generating donor banks ---")
    donor_banks = load_donor_banks(
        args.donors_dir,
        image_paths,
        bank_size=args.donor_bank_size,
        params=params,
        device=device,
    )

    if not donor_banks:
        print("ERROR: No donor banks could be generated. Check --donors-dir.")
        sys.exit(1)

    print(f"  Loaded {len(donor_banks)} donor banks: {', '.join(donor_banks.keys())}")

    # Optionally save donor banks for debugging
    donor_bank_dir = output_dir / "donor_bank"
    donor_bank_dir.mkdir(parents=True, exist_ok=True)
    for name, bank in donor_banks.items():
        np.save(str(donor_bank_dir / f"{name}.npy"), bank)
    print(f"  Saved donor banks to {donor_bank_dir}/")
    print()

    # Save config
    config_dump = {
        "images": image_paths,
        "donors_dir": args.donors_dir,
        "donor_bank_size": args.donor_bank_size,
        "mix_seed": args.mix_seed,
        "mix_same": args.mix_same,
        "mix_foreign": args.mix_foreign,
        "params": params,
    }
    with open(str(output_dir / "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    # Stage 2: Takeover fine-tuning (sequential, one letter at a time)
    print("--- Stage 2: Takeover fine-tuning ---")
    t0 = time.time()

    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        letter_output = output_dir / name
        resume = args.resume if args.resume and len(image_paths) == 1 else None

        print(f"\n--- [{i + 1}/{len(image_paths)}] Training '{name}' ---")
        train_letter(
            img_path,
            str(letter_output),
            donor_banks,
            params,
            resume=resume,
            device=device,
        )

    elapsed = time.time() - t0
    mins = elapsed / 60
    print(f"\n=== Takeover training complete: {len(image_paths)} letters in {mins:.1f} minutes ===")


if __name__ == "__main__":
    main()
