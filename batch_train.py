"""Batch training for multiple NCA characters from a YAML config.

Usage:
    uv run python batch_train.py train_config.yaml
    uv run python batch_train.py train_config.yaml --only A C E
    uv run python batch_train.py train_config.yaml --dry-run
"""

import argparse
import signal
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

import yaml

from train import DEFAULTS, train


def train_one(params, img_path, output_base, index, total):
    """Train a single character (used by both sequential and parallel paths)."""
    name = Path(img_path).stem
    char_output = f"{output_base}/{name}"
    print(f"--- [{index}/{total}] Training '{name}' from {img_path} ---")
    config = {**params, "target": img_path, "output_dir": char_output}
    train(config)
    print()


def main():
    parser = argparse.ArgumentParser(description="Batch train NCA characters from a YAML config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Train only these characters (matched against image stem, e.g. --only A C E)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be trained without running",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=argparse.SUPPRESS,  # internal flag passed to child workers
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    params = {**DEFAULTS, **(cfg.get("params") or {})}
    output_base = cfg.get("output_dir", "output")
    images = cfg.get("images", [])
    parallel = 1 if args.sequential else cfg.get("parallel", 1)

    if not images:
        print("No images specified in config.")
        sys.exit(1)

    if args.only:
        only_set = set(args.only)
        images = [img for img in images if Path(img).stem in only_set]
        if not images:
            print(f"No images matched --only {args.only}")
            sys.exit(1)

    print(f"=== NCA Batch Training ===")
    print(f"Config: {args.config}")
    print(f"Characters: {', '.join(Path(img).stem for img in images)}")
    print(f"Steps per character: {params['steps']}")
    print(f"Parallel workers: {parallel}")
    print(f"Output: {output_base}/<name>/")
    print()

    if args.dry_run:
        for img in images:
            name = Path(img).stem
            print(f"  [dry-run] {img} -> {output_base}/{name}/")
        if parallel > 1:
            chunk_size = ceil(len(images) / parallel)
            for w in range(parallel):
                chunk = images[w * chunk_size : (w + 1) * chunk_size]
                names = [Path(img).stem for img in chunk]
                print(f"  [worker {w}] {', '.join(names)}")
        return

    t0 = time.time()

    if parallel <= 1:
        # Sequential
        for i, img_path in enumerate(images):
            train_one(params, img_path, output_base, i + 1, len(images))
    else:
        # Split images into chunks and launch separate processes
        chunk_size = ceil(len(images) / parallel)
        chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

        print(f"Launching {len(chunks)} parallel workers...\n")
        procs = []
        for chunk in chunks:
            names = [Path(img).stem for img in chunk]
            cmd = [sys.executable, "batch_train.py", args.config, "--sequential", "--only"] + names
            proc = subprocess.Popen(cmd, start_new_session=False)
            procs.append((proc, names))
            print(f"  Worker PID {proc.pid}: {', '.join(names)}")

        print()

        def kill_workers(*_args):
            for proc, _ in procs:
                try:
                    proc.send_signal(signal.SIGTERM)
                except OSError:
                    pass
            sys.exit(1)

        signal.signal(signal.SIGINT, kill_workers)
        signal.signal(signal.SIGTERM, kill_workers)

        # Wait for all workers
        failed = []
        for proc, names in procs:
            proc.wait()
            if proc.returncode != 0:
                failed.append((names, proc.returncode))

        if failed:
            for names, rc in failed:
                print(f"  FAILED: {', '.join(names)} (exit code {rc})")
            sys.exit(1)

    elapsed = time.time() - t0
    mins = elapsed / 60
    print(f"=== Batch complete: {len(images)} characters in {mins:.1f} minutes ===")


if __name__ == "__main__":
    main()
