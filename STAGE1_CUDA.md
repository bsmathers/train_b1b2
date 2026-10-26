# Stage-1 B1/B2 CUDA training

The repository contains TaurosV1 Epoch475 as verified shards under
`artifacts/epoch475/`. The B1 launcher reconstructs the checkpoint automatically
under `checkpoints/` on its first run.

Requirements:

- A CUDA GPU with at least 24 GB VRAM; RTX 3090 is the cost-oriented default.
- CUDA-enabled PyTorch with BF16 support.
- Enough disk for the downloaded and extracted strong-model trajectory corpus.

Launch B1:

```bash
HF_TOKEN=optional-hugging-face-token \
WANDB_API_KEY=optional-key \
./tools/train_stage1_b1_cuda.sh
```

On first launch the script downloads the pinned `gen1ou.zip` archive from
`bsmathers/scraped_replays_parsed`, verifies its SHA-256, and safely extracts it
under `cache/strong_model_replays_parsed`. Later launches reuse that directory.
`HF_TOKEN` is optional for this public dataset. Set `GOOD_REPLAYS_DIR` to reuse
an existing local copy instead.

Defaults are batch size 12, one accumulation step, 1,000 updates per epoch,
30 epochs, a deterministic 10% held-out split, and actor-NLL early stopping
with patience 5. Override them with `BATCH_SIZE`, `GRAD_ACCUM`,
`STEPS_PER_EPOCH`, or `EPOCHS`.

The launcher defaults to eager execution because the installed AMAGO/PyTorch
combination has shown TorchDynamo failures. Standard PyTorch SDPA still uses
fused CUDA attention kernels. Set `ENABLE_TORCH_COMPILE=1` only after a short
machine-specific smoke test.
