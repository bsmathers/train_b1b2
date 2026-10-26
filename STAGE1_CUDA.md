# Stage-1 B1/B2 CUDA training

The repository contains TaurosV1 Epoch475 as verified shards under
`artifacts/epoch475/`. The B1 launcher reconstructs the checkpoint automatically
under `checkpoints/` on its first run.

Requirements:

- A CUDA GPU with at least 24 GB VRAM; RTX 3090 is the cost-oriented default.
- CUDA-enabled PyTorch with BF16 support.
- Parsed strong-model trajectories containing `gen1ou/` or `gen1ou.tar`.

Launch B1:

```bash
GOOD_REPLAYS_DIR=/path/to/strong_model_replays_parsed \
WANDB_API_KEY=optional-key \
./tools/train_stage1_b1_cuda.sh
```

Defaults are batch size 12, one accumulation step, 1,000 updates per epoch,
30 epochs, a deterministic 10% held-out split, and actor-NLL early stopping
with patience 5. Override them with `BATCH_SIZE`, `GRAD_ACCUM`,
`STEPS_PER_EPOCH`, or `EPOCHS`.

The launcher defaults to eager execution because the installed AMAGO/PyTorch
combination has shown TorchDynamo failures. Standard PyTorch SDPA still uses
fused CUDA attention kernels. Set `ENABLE_TORCH_COMPILE=1` only after a short
machine-specific smoke test.
