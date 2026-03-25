# Directional Seeding

## Question

Does NCA regeneration for this setup always happen from the center of the frame? Can I set what point to start the reset morphogenesis from?

## Answer

Yes, growth always starts from the center. The seed is a single pixel at (36, 36) with alpha + hidden channels = 1.

You can change the seed position at inference — just modify `reset()` in the viewer (or `make_seed()` in training) to place the pixel elsewhere. Since the NCA learns local rules, it will grow the same pattern from any point. However:

- **Trained at center only** — the model has only seen the seed at the center during training. Growth from a different position will work (local rules don't depend on absolute position), but the pattern will be offset. If the seed is too close to an edge, the alive mask boundary effects may clip the growth.

- **Self-repair doesn't use the seed** — when you erase part of the pattern, repair propagates inward from the surviving edges, not from a seed point. The seed is only for initial from-scratch morphogenesis.

- **To make it robust to seed position**, you'd randomize the seed location during training (and shift the target image to match). The standard paper doesn't do this.

- **Multiple seeds** also work — you can place several seed pixels and the pattern will grow from all of them simultaneously (though it was trained for single-seed growth, so results may be unpredictable).
