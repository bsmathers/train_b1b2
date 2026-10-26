# Stage-1 B1 CUDA training

The B1 launcher reconstructs the sharded Epoch475 checkpoint, installs missing
Python dependencies, and downloads the required Gen1 OU datasets:

- 50% curated strong-model trajectories from
  `bsmathers/scraped_replays_parsed`;
- 40% human parsed trajectories from `jakegrigsby/metamon-parsed-replays`;
- 10% `pac-tauros` from `jakegrigsby/metamon-parsed-pile`.

`HF_TOKEN` is optional for these public datasets. Downloads and extracted data
are reused from `cache/` on later launches.

```bash
HF_TOKEN=optional-hugging-face-token \
WANDB_API_KEY=optional-key \
./tools/train_stage1_b1_cuda.sh
```

The old four-critic offline objective is gone. B1 now has the Epoch475 actor and
backbone plus one categorical loss/draw/win value head. Every replay state is
given its complete-trajectory W/L/D label. Epoch475 policy probabilities are
computed lazily from a frozen teacher during training rather than materialized
as a multi-gigabyte duplicate dataset.

Training runs three stages:

1. value head only, up to 100 epochs, early-stopping patience 3;
2. value head plus full backbone, actor head frozen, with Epoch475 policy KL
   and backbone L2-SP, up to 1,000 epochs, patience 5;
3. actor, value, and full backbone with advantage-weighted imitation, Epoch475
   policy KL, stage-2 value-distribution KL, and stage-2 backbone L2-SP, up to
   1,000 epochs, patience 5.

Each phase validates its entry model before taking an update, and early stopping
restores matching policy, AdamW, and learning-rate-scheduler state. Validation
uses a fresh deterministic sample from the held-out split on every pass rather
than repeatedly overfitting the same 600 trajectories.

L2-SP weights are configurable by component-aligned parameter-name prefix in
`stage1_b1_imitation.gin`; the longest matching prefix wins. For example,
`"traj_encoder": 2.0` can protect the trajectory transformer twice as strongly,
while a more specific `"traj_encoder.tformer.layers.0": 4.0` protects one block.
Weights are normalized by weighted parameter count, and `0.0` excludes a layer.

An epoch defaults to 100 optimizer updates. Best phase checkpoints are written
under `checkpoints/stage1_b1/ckpts/policy_weights/` as:

- `policy_value_head_best.pt`
- `policy_value_backbone_best.pt`
- `policy_actor_best.pt`

The final restored actor is also published as `policy_final_best.pt`, a new
numbered `policy_epoch_N.pt`, and `ckpts/latest/policy.pt`, so standard evaluation
tools cannot accidentally load the final rejected epoch.

The launcher defaults to BF16, eager execution, and fused FlashAttention with
the same causal `(32, 0)` window as Epoch475. It installs and smoke-tests
`flash-attn` when needed. Override `BATCH_SIZE`,
`STEPS_PER_EPOCH`, `DLOADER_WORKERS`, or `MIXED_PRECISION` as needed.
