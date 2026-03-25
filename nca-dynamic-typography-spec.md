# NCA Dynamic Typography — Project Spec

## Project Goal

Build a dynamic typography system using Neural Cellular Automata (NCA) where handwritten characters are rendered as living, self-repairing textures. Characters can be damaged, shifted, and perturbed at the pixel level to create organic motion and transitions, then allowed to regenerate back to their target states. The result is typography that moves and breathes with interesting pixel-level dynamics — more alive than conventional motion graphics, without requiring hand-animated frames.

## Motivation

Standard procedural typography techniques produce motion that is either mechanically smooth or relies on pre-authored keyframes. NCA offers a fundamentally different approach: the motion and texture emerge from learned local rules, producing organic, never-exactly-the-same-twice behavior. The NCA's damage repair property is the core creative mechanism — perturbation creates decoherence, removal of perturbation allows recovery, and the transition between those states is the animation.

Key aesthetic goals:
- Characters should decohere **gradually** under sustained perturbation, not glitch or snap
- Characters should **reliably return** to their target state once perturbation stops, even from severe damage
- The pixel-level motion during repair should feel organic — like ink flowing back into place, not a digital undo
- Perfect fidelity is not required; slight NCA wobble and imperfection suit the handwritten aesthetic

## Architecture Overview

The system has two distinct phases:

1. **Training (offline, PyTorch on Apple Silicon):** Train NCA models to grow and maintain handwritten characters with damage repair. Export learned weights as flat float32 arrays.
2. **Inference + Visualization (realtime, WebGPU):** Run trained NCA update rules in compute shaders. Apply perturbation (grid shifts, noise injection, selective damage) at runtime to create motion and transitions. Render visible channels to screen.

These two phases share no code. The bridge between them is a small weights file (~30–50KB per character) containing the learned 1×1 conv parameters.

---

## Training Setup

### Model Architecture

This follows the original "Growing Neural Cellular Automata" (Mordvintsev et al., 2020) recipe exactly. It is the most widely reproduced NCA configuration and converges reliably with minimal hyperparameter tuning.

- **Grid size:** 72×72 (40×40 target centered with 16px padding on each side)
- **Channels:** 16 total — first 4 are RGBA (visible), remaining 12 are hidden state
- **Perception step:** Fixed (not learned) depthwise 3×3 convolutions using hardcoded kernels:
  - Identity filter (passes through current state)
  - Sobel-X filter (horizontal gradient)
  - Sobel-Y filter (vertical gradient)
  - Output: 48-dimensional perception vector per cell (16 channels × 3 filters)
- **Update network:** Two 1×1 conv layers
  - `perception (48) → hidden (128, ReLU) → output (16)`
  - **Critical:** Initialize the final layer weights to zero ("do nothing" initialization). This ensures the system starts stable and gradually learns to act.
- **Stochastic update mask:** Each cell has an independent 50% probability of applying its update at each step. This prevents dependence on a global clock and is essential for robust self-repair.
- **Alive masking:** A cell is "alive" if any cell in its 3×3 neighborhood has alpha > 0.1. Dead cells get their entire state (all 16 channels) zeroed. This prevents unbounded growth.
- **Residual update:** The network output is *added* to the current cell state (not replaced). The network learns incremental deltas.
- **Total parameters:** ~8,000

### Target Images

- 40×40 RGBA images on transparent background
- One image per character (handwritten A–Z, possibly digits and punctuation)
- **Stroke width should be 4–6 pixels**, not hairline thin. Thin strokes (1–2px) lack spatial redundancy for reliable repair and dissolve abruptly under perturbation. Thicker strokes decohere more gracefully. Think "marker on paper" not "fine pen."
- Source: render from a handwriting font, or hand-draw and scan. The organic imperfection of real handwriting plays well with NCA dynamics.

### Training Loop

- **Pool-based training:**
  - Maintain a pool of 1024 grid states (partially grown / fully formed / previously damaged)
  - Each iteration: sample a batch of 8 from the pool
  - Run the NCA forward for a random number of steps sampled uniformly from [64, 96]
  - Compute loss, backprop, update weights
  - Write the batch's final states back into the pool (replacing the sampled entries)
- **Loss function:** L2 (MSE) on the 4 RGBA channels only, compared against the target image, averaged over all cells. Do not include hidden channels in the loss.
- **Optimizer:** Adam, learning rate 2e-3
- **Damage training (include from the start, not as a second phase):**
  - Each batch, find the pool entry with the highest loss (most degraded state)
  - Replace it with a damaged version: zero out a random rectangular region covering 25–50% of the alive cells (zero all 16 channels, not just RGBA)
  - This forces the network to learn repair as part of normal training, not as a separate objective

### Training Expectations

- **Convergence timeline:** Pattern recognizable by ~2,000–3,000 steps. Sharp and stable with good repair by 8,000–10,000 steps.
- **Wall clock time per character:**
  - M1 Max (32-core GPU, 32GB): ~15–20 minutes
  - M4 base (10-core GPU, 16GB): ~20–30 minutes
- **Full alphabet (26 characters, sequential):** ~9 hours on M1 Max, ~13 hours on M4. Run overnight.
- **Hardware note:** The NCA model is tiny (~8K params) and the grid is small (72×72×16). Neither machine will be compute-bound or memory-bound. The training bottleneck is backpropagation through 64–96 sequential NCA steps, which is inherently serial. Both machines use the same hyperparameters with no changes.

### Long-Term Stability

If the pattern drifts or oscillates after hundreds of steps at inference time, increase the max step count during training from 96 to 128 or 192. This forces the network to maintain the pattern over longer horizons. The pool-based training already helps significantly with this because old pool entries have implicitly run for many cumulative steps.

### Weight Export

After training, export the model weights as raw float32 arrays in a simple format (JSON with base64-encoded weight blobs, or a flat binary file). The data to export per character:

- Update layer 1: weight tensor shape `[128, 48, 1, 1]` + bias `[128]`
- Update layer 2: weight tensor shape `[16, 128, 1, 1]` + bias `[16]`
- Total: ~6,300 floats → ~25KB per character

The Sobel perception kernels are hardcoded (not learned), so they don't need to be exported — they'll be hardcoded in WGSL as well.

---

## Inference / Visualization Setup (WebGPU)

### Compute Shader Structure

Each frame runs one or more NCA update steps as a compute shader dispatch. Each cell is one invocation. The shader:

1. **Perturbation pass (optional):** Before the NCA update, apply any runtime perturbation — grid shift, noise injection, selective zeroing. This is the creative control surface.
2. **Perception:** Read the 3×3 neighborhood for the current cell. Apply the three fixed Sobel kernels to produce the 48-dim perception vector.
3. **Update:** Matrix multiply perception through the two dense layers (weights loaded from storage buffers), with ReLU between them.
4. **Stochastic mask:** Generate a per-cell random bit. If 0, skip the update for this cell (output = current state unchanged).
5. **Alive mask:** Check if any neighbor has alpha > 0.1. If not, zero the entire cell state.
6. **Residual add:** Add the network output to the current state (only for cells that passed the stochastic mask).

Use a **ping-pong buffer** pattern: two state textures (or storage buffers) of shape `[72, 72, 16]`. Read from buffer A, write to buffer B, swap each step.

A separate render pass reads the first 4 channels (RGBA) from the current state buffer and draws to a fullscreen quad (or samples for each character's tile if compositing multiple letters).

### Primary Perturbation Technique: Grid Shifting

The core motion technique is translating cell states by copying them one pixel in a direction, then letting the NCA repair the disrupted edges.

Implementation: a compute pass before the NCA update that copies `state[y][x] = state[y][x + dx]` (and `state[y][x] = state[y + dy][x]` for vertical shift). The vacated edge column/row is filled with zeros.

**Critical:** Shift *all 16 channels*, not just RGBA. The hidden channels carry the distributed repair information. Shifting only visible channels will produce garbage at the edges.

Key parameters to expose as runtime controls:
- **Shift rate:** How many NCA steps between each pixel shift. Sweet spot is likely 3–6 steps. 1 step = fast/ragged, 10+ steps = smooth/crisp. Put this behind a slider.
- **Shift direction:** Can change over time for curved paths. Shift X every N steps and Y every M steps for diagonal motion.
- **Shift pulsing:** Shift for K frames, pause for J frames. Creates natural start-stop locomotion rhythm and gives the NCA recovery windows.

### Additional Perturbation Techniques to Explore

These require no retraining — they're all runtime manipulations of cell state:

- **Selective rectangular damage:** Zero out a region of cells. The NCA regrows it. Useful for "erasing" a letter and watching it return.
- **Noise injection into hidden channels:** Add small random values to channels 4–15. Causes visible wobble/shimmer without destroying the pattern. Intensity controls how psychedelic it gets.
- **Channel-specific perturbation:** Different hidden channels encode different aspects of the pattern. Perturbing specific channels may cause color shifts or structural distortions. This is exploratory — the channels aren't interpretable by default.
- **Alpha channel manipulation:** Zeroing just the alpha of some cells triggers the alive mask cascade — neighboring cells notice the "death" and attempt repair, creating organic reveal/dissolve effects.
- **Seed-based regrowth:** Zero the entire grid, place a single seed pixel with alpha=1 and all other channels at some initial value. The character regrows from scratch. The growth animation is itself a compelling visual.

### Multi-Character Compositing (Spelling Words)

**Approach: One NCA instance per character, composited at render time.**

Each letter runs its own independent NCA on its own 72×72 grid with its own weights. The WebGPU renderer samples from each grid and places them side by side (or overlapping) to spell words. The letters don't share a grid and don't interact at the NCA level — interaction effects (if desired) happen at the render compositing level.

This is simpler and more reliable than training a single conditioned NCA for all characters. It also means each letter can be independently perturbed — shift one letter while others stay still, damage only the middle letter, reveal letters one at a time.

**Future upgrade path:** Train a single NCA conditioned on a "genome" signal that specifies which character to produce. This would unlock smooth morphing between letters (R dissolving into S through intermediate forms) and true multi-character coexistence on a shared grid. However, this is significantly harder to train reliably and should be attempted only after the base per-character approach is working.

---

## Development Phases

### Phase 1: Single Character End-to-End
- Train one handwritten character (pick one with interesting stroke structure — "R", "g", or "k" are good candidates)
- Confirm convergence with the exact hyperparameters above
- Export weights
- Build minimal WebGPU visualizer: NCA update shader + fullscreen render
- Verify that damage repair works at runtime (click to damage, watch it regrow)
- **Goal: working pipeline from training to realtime visualization**

### Phase 2: Motion via Grid Shifting
- Implement the grid shift compute pass
- Add runtime controls for shift rate, direction, and pulsing
- Tune the shift rate vs repair rate balance for the handwritten character aesthetic
- **Goal: one letter that drifts across screen with organic trailing-edge repair**

### Phase 3: Full Alphabet + Word Compositing
- Train all 26 characters (same hyperparameters, swap target image, automate batch training)
- Build the multi-instance compositor in WebGPU
- Spell words with independently controllable letters
- **Goal: dynamic typography that spells actual words**

### Phase 4: Creative Perturbation Exploration
- Implement additional perturbation modes (noise injection, selective damage, seed regrowth)
- Build a control surface: which perturbation, intensity, spatial targeting
- Experiment with choreographed sequences: letters revealing one by one, words dissolving and reforming, audio-reactive perturbation intensity
- **Goal: a creative tool / performance instrument for NCA typography**

---

## Key References

- Mordvintsev et al., "Growing Neural Cellular Automata" (2020) — Distill.pub. The foundational paper. Use this as the primary implementation reference. Includes runnable Colab notebooks.
- Niklasson et al., "Self-Organising Textures" (2021) — Distill.pub. Texture synthesis extension (not needed for this project, but relevant background).
- Pajouheshgar et al., "Mesh Neural Cellular Automata" (2024) — ACM TOG. Introduces grafting/interpolation between trained NCAs.
- Multi-texture NCA via genomic signals — Scientific Reports (2025). Single NCA producing multiple textures via conditioning. Relevant for the future conditioned multi-character approach.

## Dependencies

### Training
- `torch` (with MPS backend for Apple Silicon)
- `numpy`
- `Pillow` (target image loading)
- `matplotlib` (optional, for training visualization)

### Inference
- Vanilla WebGPU API (no libraries)
- WGSL shaders (~80–120 lines for the NCA update kernel)
