# Project (Runtime Workspace)

Think of `project/` as Onika’s **working directory** for training.

It’s where run-specific stuff tends to land (datasets you’re currently using, caches, outputs, debug logs).

In practice:
- You prepare your dataset under `project/dataset/`
- Training will produce caches under `project/cache/`
- Results (LoRAs, samples, logs) show up under `project/outputs/`
- Optional regularization images can live in `project/reg/`
