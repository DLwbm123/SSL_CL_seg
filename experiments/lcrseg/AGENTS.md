# LCR-Seg Engineering Rules

- Treat `METHOD_SPEC_V0_1.md` as the method source of truth.
- Do not change LCR-Seg equations, tensor contracts, detach rules, or lifecycle semantics without updating all three specification files and their tests.
- Do not implement V4 multi-agent, V5 RIC, `K > 1`, replay, diffusion, VAE, a third teacher, or additional auxiliary losses in V0.1.
- Reuse one training engine for all baselines and LCR-Seg; do not duplicate a training loop for a new method.
- The old model and historical anchors must never receive gradients or updates.
- Anchor buffers must never be optimizer parameters.
- Detach weak pseudo-labels, learnability maps, and compatibility maps used as weights.
- Hidden diagnostic labels must never enter training loaders or training configuration.
- Run unit, one-batch backward, golden-batch, checkpoint-resume, and two-case overfit tests at every milestone.
- Stop and report a BLOCKER rather than inventing data semantics, paths, or method definitions.
- Do not modify frozen HDF5, manifests, splits, or checksums.
- Do not push automatically.
