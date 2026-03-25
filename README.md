# NCA Dynamic Typography

Neural Cellular Automata training pipeline for handwritten characters with self-repair. Based on [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev et al., 2020).

Train NCA models that grow and maintain characters from a single seed pixel, then export weights for realtime WebGPU inference with perturbation effects (grid shifting, damage, noise).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs PyTorch with MPS (Metal) support for Apple Silicon. No special index needed.

## Training

Provide a 40x40 RGBA PNG on a transparent background. Stroke width should be 4-6 pixels.

```bash
# Train on a target image (outputs to output/<name>/):
uv run python train.py --target images/A.png

# Generate a test character and train on it:
uv run python train.py --char R

# Custom step count and output directory:
uv run python train.py --target images/A.png --steps 10000 --output-dir output/A_long

# Resume from a checkpoint:
uv run python train.py --target images/A.png --resume output/A/checkpoint_004000.pt
```

Full training (8000 steps) takes ~15-20 minutes per character on M1 Max.

## Monitoring results

Training progress is printed to the terminal every 100 steps. Outputs are saved to the output directory:

```
output/A/
  vis/                  # Grid images every 500 steps (NCA state vs target)
  loss.png              # Loss curve (saved at end)
  checkpoint_*.pt       # Model checkpoints every 2000 steps
  checkpoint_final.pt   # Final model weights
  weights.json          # Exported weights for WebGPU (base64-encoded float32)
  weights.bin           # Exported weights for WebGPU (raw binary)
```

The `vis/` directory is the main place to check -- it shows the batch of NCA states composited over white alongside the target image.

## Architecture

- **Grid:** 72x72 (40x40 target + 16px padding), 16 channels (4 RGBA + 12 hidden)
- **Perception:** Fixed depthwise 3x3 convolutions (identity + Sobel-X + Sobel-Y)
- **Update network:** Two 1x1 conv layers (48 -> 128 -> 16), ~8,300 parameters
- **Training:** Pool-based (1024 states), batch 8, 64-96 NCA steps per iteration, circular damage on best samples

See [nca-dynamic-typography-spec.md](nca-dynamic-typography-spec.md) for the full project spec including WebGPU inference and perturbation techniques.
