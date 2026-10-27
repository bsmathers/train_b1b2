# Stage-1 B1 CUDA training

The B1 launcher reconstructs the sharded Epoch475 checkpoint, installs missing
Python dependencies, and downloads the required Gen1 OU datasets:

- 50% curated strong-model trajectories from
  `bsmathers/scraped_replays_parsed`;
- 40% human parsed trajectories from `jakegrigsby/metamon-parsed-replays`;
- 10% `pac-tauros` from `jakegrigsby/metamon-parsed-pile`.

For low-disk machines, the launcher processes pac-tauros first and streams its
compressed tar directly from Hugging Face. It retains a deterministic uniform
reservoir of 250,000 trajectories and never writes the full decompressed tar.
The dataset's training mixture weight remains 10%; only its on-disk diversity is
bounded. Set `PAC_TAUROS_TRAJECTORIES` to change this bound. Completed
strong-replay archives are also removed after extraction.

Policy-only numbered checkpoints default to every 10 epochs to keep a long run
within the disk budget; the redundant AMAGO Accelerate-state history is disabled
for this non-resumable staged pipeline. Validation-best snapshots still save
matching policy, optimizer, and scheduler state immediately. Set `CKPT_INTERVAL`
to override the numbered-policy interval.

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

Value training runs two stages:

1. value head only, up to 100 epochs, early-stopping patience 3;
2. value head plus full backbone, actor head frozen, with Epoch475 policy KL
   and backbone L2-SP, up to 1,000 epochs, patience 5;
Each phase validates its entry model before taking an update, and early stopping
restores matching policy, AdamW, and learning-rate-scheduler state. Validation
uses a fresh deterministic sample from the held-out split on every pass rather
than repeatedly overfitting the same 600 trajectories.

L2-SP weights are configurable by component-aligned parameter-name prefix in
`stage1_b1_imitation.gin`; the longest matching prefix wins. For example,
`"traj_encoder": 2.0` can protect the trajectory transformer twice as strongly,
while a more specific `"traj_encoder.tformer.layers.0": 4.0` protects one block.
Weights are normalized by weighted parameter count, and `0.0` excludes a layer.

An epoch defaults to 1,000 optimizer updates. Best phase checkpoints are written
under `checkpoints/stage1_b1/ckpts/policy_weights/` as:

- `policy_value_head_best.pt`
- `policy_value_backbone_best.pt`
- `policy_phase2_final.pt`

The final restored Phase-2 model is published as `policy_phase2_final.pt`. A
separate strong-only Epoch20-style refinement can be launched with
`tools/train_stage1_b1_actor_refine_cuda.sh`. The refinement uses only strong
model trajectories, plain action cross-entropy, Epoch475 policy KL, and a
Phase-2 value-output KL. The value head is frozen. It uses a fresh optimizer,
an effective batch of 48 by default, and a fixed held-out sample of 6,000
sequences. Its restored best actor is saved as `policy_actor_bc_final.pt`, a numbered
`policy_epoch_N.pt`, and `ckpts/latest/policy.pt`, so standard evaluation tools
cannot accidentally load the final rejected epoch.

The launcher defaults to BF16, eager execution, and fused FlashAttention with
the same causal `(32, 0)` window as Epoch475. It installs and smoke-tests
`flash-attn` when needed. Override `BATCH_SIZE`,
`STEPS_PER_EPOCH`, `DLOADER_WORKERS`, or `MIXED_PRECISION` as needed.
