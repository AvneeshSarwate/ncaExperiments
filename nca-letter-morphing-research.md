# NCA Letter Morphing Research

Created: 2026-03-30

This note has two purposes:

1. Preserve the human-readable synthesis from the initial research pass.
2. Provide a broader research index so a fresh agent run can recover context quickly.

The first report below is kept as-is from the earlier response. The supplemental index at the end includes additional sources I opened or surfaced during research, plus the local repo/code references I actually inspected.

## Initial Report

**Core Takeaway**

Your live weight-swap failed for the expected reason: the UI explicitly preserves the current NCA state when switching models in [main.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/main.ts#L47), and `setModel()` in [nca-viewer.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/nca-viewer.ts#L157) only swaps the rule weights, not the hidden-state semantics. Independently trained letter models do not share a compatible hidden space, so model `B` sees model `A`’s latent channels as garbage. For morphing, you almost certainly want one shared model, or a learned state adapter, not runtime weight swapping.

There is also a repo-level issue to fix before serious morph work: training chooses a target-specific seed near each glyph centroid in [train.py](/Users/avneeshsarwate/agentCombine/ncaTest/train.py#L111) and [train.py](/Users/avneeshsarwate/agentCombine/ncaTest/train.py#L363), while the viewer always resets at the center in [nca-viewer.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/nca-viewer.ts#L172). Cross-letter morphing wants a single canonical alignment.

**Research Map**

- Best-fit family: one shared conditional NCA with a persistent target signal. The paper you linked, [Conditional Morphogenesis](https://arxiv.org/abs/2512.08360), shows a single NCA can grow many classes from one seed by injecting a broadcast class vector every step. Important nuance: it demonstrates multi-target growth, not mid-trajectory `A -> B` switching by itself.
- Most directly relevant morphing paper: [Goal-Guided Neural Cellular Automata](https://openreview.net/pdf?id=rNnlEHG16b5). They explicitly train one NCA to morph between emojis under changing goal encodings and report stable mappings plus interpolation between goals.
- Signal-triggered switching: [Neural Cellular Automata Can Respond to Signals](https://eprints.lancs.ac.uk/215344/4/isal_a_00567.pdf). This is useful if you want a one-pixel or transient steering cue. Their key lesson is important: if you only train one-way change, the system may effectively replace itself once and then lose the ability to respond again; repeated switching during training fixes that.
- Seed/genome conditioning: [StampCA](https://kvfrans.com/stampca-conditional-neural-cellular-automata/) puts the condition in the seed state instead of broadcasting it every step. Good for “DNA-like” behavior, but weaker for runtime retargeting unless you also train a way to rewrite that genome.
- Weight-manifold / hypernetwork approaches: [Neural Cellular Automata Manifold](https://arxiv.org/abs/2006.12155) and [Variational Neural Cellular Automata](https://arxiv.org/abs/2201.12360) learn spaces of NCA behaviors rather than one fixed rule. This is promising if you want continuous interpolation between letters, but it is more engineering-heavy than direct conditioning.
- Local rule-mixture approaches: [Mixtures of Neural Cellular Automata](https://arxiv.org/abs/2506.20486) and [MeshNCA](https://meshnca.github.io/) point toward grafting / mixed local behaviors. This is more speculative for letters, but it is a real direction if you want region-by-region takeover rather than a single global retarget.

**What Peter Whidden / The Cloned Repos Suggest**

- Peter Whidden’s repo is directly relevant. In [basic_large.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Basic/basic_large.py#L126), he uses control channels inside the CA state to steer one shared model toward different outputs. That is the right conceptual replacement for your current multi-weights UI.
- The same repo also has a direct image-to-image setup in [ca_img2img_v1.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Img2Img/ca_img2img_v1.py#L104). That suggests another path: start from a mature `A` image and train the NCA to converge to `B`, instead of insisting on seed growth for every transition.
- In `cloned_ref_repos`, the only non-PyTorch repo is `utnca-paint`. Its useful idea is not “morph one object into another,” but “use a control field to select among local CA rules.” You can see that in [fragment.glsl](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/utnca-paint/src/fragment.glsl#L235). That is good inspiration for takeover / grafting / local transition waves.

**What I’d Actually Try Here**

- Highest-probability path: replace per-letter models with one conditional NCA. Feed a target-letter embedding every step. In WebGPU, the simplest implementation is one shared `weights.bin` plus a small condition buffer or embedding table, not 26 separate models.
- Train with three state types in one pool: fresh seeds, stable same-letter states, and stable other-letter states. The third category is what teaches real morphing instead of just repair.
- Add explicit switch curriculum: run under target `A` for some steps, switch to target `B`, then optimize toward `B`. Sample all ordered pairs. `N^2` pair sampling inside one model is reasonable for 26 letters; `N^2` separate models is not.
- On switch, do not preserve the full hidden state blindly. The simplest strong baseline is: keep RGBA/alpha, zero hidden channels, then let the shared conditional model regrow toward `B`. More advanced version: partial hidden reset or learned reset gate.
- Train repeated toggles, not one-off transitions. `A -> B -> A` matters. The signals paper is a warning here: otherwise you can get a one-time takeover that does not remain steerable.
- If you care about smooth-looking morphs rather than “erase and regrow,” add an intermediate-shape curriculum. The practical hack is SDF interpolation or a common scaffold phase, so the model learns a path instead of only the endpoint.
- Normalize all glyphs to a common coordinate frame and use a consistent seed policy. Your current target-dependent seed location will make the model learn translation as part of the transition.

**Hacky Solutions That Are Actually Reasonable**

- Common-scaffold trick: train every letter to pass through the same blob / skeleton / ring before specializing. Then morphs become `A -> scaffold -> B`, which is much easier than learning 650 unrelated bridges.
- Foreign-state takeover training: for each target letter `B`, train from damaged states and from other letters’ visible states, not just from seed. This can make “become B from arbitrary garbage” work surprisingly well, especially if you reset hidden channels first.
- Learned hidden-state adapter: keep separate letter models, but train a tiny per-pair `1x1` state adapter `A_state -> B_state` before swapping weights. I would treat this as a quick art hack, not the long-term architecture.
- Local takeover / grafting: inspired by `utnca-paint`, inject a “B-rule” signal in one region and let it spread. This is more experimental, but it could produce interesting biological-looking letter conversions.
- Pairwise separate models: only worth it if this is a one-off demo and you do not care about scale, generalization, or browser complexity.

If you want one concrete recommendation: stop training separate letter weights and build a single conditional model with a switch curriculum and hidden-state reset-on-switch. That is the cleanest path from your current setup to actual runtime `A -> B` morphing.

**Sources**

- [Conditional Morphogenesis: Emergent Generation of Structural Digits via Neural Cellular Automata](https://arxiv.org/abs/2512.08360)
- [Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems](https://openreview.net/pdf?id=rNnlEHG16b5)
- [Neural Cellular Automata Can Respond to Signals](https://eprints.lancs.ac.uk/215344/4/isal_a_00567.pdf)
- [StampCA: Growing Emoji with Conditional Neural Cellular Automata](https://kvfrans.com/stampca-conditional-neural-cellular-automata/)
- [Peter Whidden / pwhiddy NCA repo](https://github.com/pwhiddy/Growing-Neural-Cellular-Automata-Pytorch)
- [Neural Cellular Automata Manifold](https://arxiv.org/abs/2006.12155)
- [Variational Neural Cellular Automata](https://arxiv.org/abs/2201.12360)
- [Mixtures of Neural Cellular Automata](https://arxiv.org/abs/2506.20486)
- [Mesh Neural Cellular Automata](https://meshnca.github.io/)

## Additional Research Index

This section covers sources that were looked at or surfaced during research but were not included in the shorter source list above. I am including both external links and the local code files that were directly inspected.

I did not index obviously irrelevant search noise. The list below focuses on sources that were at least plausibly relevant to NCA morphing, steering, takeover, conditioning, or repo implementation.

### Additional External Sources Opened Or Intentionally Inspected

- [Conditional-NCA-Digits GitHub repo](https://github.com/alisakour/Conditional-NCA-Digits) - code repo linked from the `2512.08360` paper.
- [Conditional Morphogenesis ar5iv HTML mirror](https://ar5iv.labs.arxiv.org/html/2512.08360) - easier-to-scan HTML rendering of the same paper.
- [shyamsn97/controllable-ncas](https://github.com/shyamsn97/controllable-ncas) - code for GoalNCA, includes explicit `Morphing Emoji` training setup and shows a practical goal-encoder implementation.
- [A Path to Universal Neural Cellular Automata](https://arxiv.org/abs/2505.13058) - follow-on work referenced from Peter Whidden’s repo.
- [Peter Whidden transdimensional demo link](http://transdimensional.xyz/projects/neural_ca/index.html) - runtime demo link referenced from the repo README.
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) - important background on dynamic NCA behavior and multi-target / multi-style texture synthesis.
- [Multi-texture synthesis through signal responsive neural cellular automata](https://www.nature.com/articles/s41598-025-23997-7) - signal-responsive / genome-channel texture-control paper that surfaced during search.
- [Multi-texture synthesis through signal responsive neural cellular automata (PMC mirror)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12623773/) - open-access version of the same article.
- [Adversarial Takeover of Neural Cellular Automata project page](https://letteraunica.github.io/neural_cellular_automata/) - directly relevant to takeover / reprogramming ideas.
- [Adversarial Takeover of Neural Cellular Automata PDF](https://letteraunica.github.io/neural_cellular_automata/paper.pdf) - paper PDF used as a more direct source than third-party summaries.
- [ALIFE 2022 volume page for Adversarial Takeover of Neural Cellular Automata](https://direct.mit.edu/isal/isal2022/volume/34) - official proceedings index page where the paper appears.
- [awesome-neural-cellular-automata](https://github.com/dwoiwode/awesome-neural-cellular-automata) - literature hub that surfaced during search and is useful for follow-on exploration.

### Additional External Leads Surfaced But Not Used Heavily

- [Michael Levin computational publications page](https://drmichaellevin.org/publications/computational.html) - surfaced while tracing takeover / reprogramming references.
- [Tufts Levin Lab publications](https://as.tufts.edu/biology/levin-lab/publications) - another bibliographic trail for takeover / reprogramming work.
- [Peter Whidden resume PDF](https://transdimensional.xyz/PeterWhiddenResume.pdf) - surfaced during search but not important for the technical conclusions.

### Local Repo References Reviewed In This Repo

- [train.py](/Users/avneeshsarwate/agentCombine/ncaTest/train.py#L111) - `find_seed_position`, which places the seed near the target centroid.
- [train.py](/Users/avneeshsarwate/agentCombine/ncaTest/train.py#L363) - seed creation and pool initialization using the target-specific seed.
- [nca_model.py](/Users/avneeshsarwate/agentCombine/ncaTest/nca_model.py#L17) - current NCA architecture and hidden-state semantics.
- [drawing-tool/src/main.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/main.ts#L47) - UI text and behavior noting that switching models preserves the current NCA state.
- [drawing-tool/src/main.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/main.ts#L290) - `modelSelect` handler that calls `viewer.setModel(...)` without resetting state.
- [drawing-tool/src/nca-viewer.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/nca-viewer.ts#L157) - `setModel()` swaps weights only.
- [drawing-tool/src/nca-viewer.ts](/Users/avneeshsarwate/agentCombine/ncaTest/drawing-tool/src/nca-viewer.ts#L172) - `reset()` always seeds the center pixel.
- [nca-webgpu/nca-engine.ts](/Users/avneeshsarwate/agentCombine/ncaTest/nca-webgpu/nca-engine.ts) - reusable WebGPU engine code for inference and validation.

### Local References Reviewed In Peter Whidden’s PyTorch Repo

- [cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/README.md](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/README.md) - repo overview and statement that a single network can converge to multiple target outputs via control channels.
- [cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Basic/basic_large.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Basic/basic_large.py#L126) - control-channel writing into the shared state and runtime control switching.
- [cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Img2Img/cifar_dataset.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Img2Img/cifar_dataset.py) - image-to-image conditioning setup with label strip / input-target construction.
- [cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Img2Img/ca_img2img_v1.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/Growing-Neural-Cellular-Automata-Pytorch/CA_Img2Img/ca_img2img_v1.py#L104) - image-to-image NCA initialization from an existing image instead of a seed.

### Local References Reviewed In The Non-PyTorch `utnca-paint` Repo

- [cloned_ref_repos/utnca-paint/README.md](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/utnca-paint/README.md) - points to the associated paper / project context.
- [cloned_ref_repos/utnca-paint/training/ca.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/utnca-paint/training/ca.py) - compact CA rule parameterization and GLSL export.
- [cloned_ref_repos/utnca-paint/training/train.py](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/utnca-paint/training/train.py#L112) - long-horizon evaluation and GLSL export path.
- [cloned_ref_repos/utnca-paint/src/fragment.glsl](/Users/avneeshsarwate/agentCombine/ncaTest/cloned_ref_repos/utnca-paint/src/fragment.glsl#L235) - painted per-pixel rule selection among many trained CA rules.

### Quick Retrieval Notes For A Fresh Agent

- The most actionable external codebase for morphing is `controllable-ncas`.
- The most actionable local idea from the cloned repos is Peter Whidden’s control-channel approach.
- The most actionable local idea from the non-PyTorch repo is `utnca-paint`’s spatial rule-selection field.
- The biggest current blocker in this repo is that the UI swaps weights while preserving incompatible hidden state.
- The second blocker is inconsistent seeding and alignment between training and WebGPU inference.
