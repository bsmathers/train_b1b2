# Ataraxos native self-play infrastructure

This is the native collection, conservative policy/value update, and sequential
round-orchestration layer for the next online stage.

## What is implemented

- Battles use the repository's native Zig `@pkmn/engine` adapter, not a
  Pokemon Showdown websocket server.
- Both players use batched transformer inference and independent persistent
  RL2/KV state. Self-play shares the read-only policy weights in GPU memory.
- Each recorded decision contains the committed action, exact behavior log
  probabilities, legal-action mask, scalar W/L value, and immutable behavior
  checkpoint identity.
- Only decisive games enter an Ataraxos round. Time-limited or otherwise
  unresolved games are excluded rather than counted as losses.
- Replay compression runs in a bounded writer pool and is flushed before a
  round is validated.
- A completed round is rejected if probability mass, legality, action
  alignment, outcomes, or behavior-policy identity are inconsistent.
- Physical player side alternates across rounds by default.
- Round summaries and progress logs are plain text.
- The reference learner replays complete recurrent trajectories, verifies the
  pre-update policy against the exact stored behavior probabilities, and then
  applies clipped PPO plus categorical W/L/D value learning.
- Checkpoints and round/update summaries are published atomically.

## One collection round

On a provisioned CUDA host:

```bash
cd /workspace/train_b1b2
PYTHON_BIN=/workspace/b1-venv/bin/python \
METAMON_CACHE_DIR=/workspace/b1-data/cache \
SAVE_DIR=/workspace/b1-data/checkpoints \
CHECKPOINT=/workspace/b1-data/checkpoints/stage1_b1_epoch51_selected_actor/ckpts/policy_weights/policy_actor_bc_final.pt \
OUTPUT_DIR=/workspace/b1-data/ataraxos \
GAMES=2000 LANES=64 SIM_WORKERS=4 \
./tools/run_ataraxos_cuda.sh
```

The round can be inspected without loading a model:

```bash
/workspace/b1-venv/bin/python -m metamon.rl.ataraxos \
  --mode inspect \
  --output-dir /workspace/b1-data/ataraxos \
  --round 0
```

Rerunning an interrupted round validates the existing files, verifies that
their behavior-policy hash matches, and collects only the missing decisive
games.

Set `PROFILE=1` to emit the collector and adapter timing breakdown. Profiling
results and tuning guidance are in `docs/ATARAXOS_PERFORMANCE.md`.

## Update hook contract

Pipeline mode is sequential: collect a round, run one update command, then use
the returned checkpoint for the next round. It does not create a scheduler or
background service.

The update process receives:

```text
ATARAXOS_ROUND_DIR
ATARAXOS_REPLAY_DIR
ATARAXOS_BEHAVIOR_CHECKPOINT
ATARAXOS_PARENT_CHECKPOINT
ATARAXOS_NEXT_CHECKPOINT_FILE
ATARAXOS_ARCHITECTURE_SAVE_ROOT
ATARAXOS_ARCHITECTURE_RUN_NAME
ATARAXOS_BEHAVIOR_TEMPERATURE
```

`ATARAXOS_BEHAVIOR_CHECKPOINT` is the immutable content-addressed copy that
actually generated the round. The updater must atomically publish its new
checkpoint, then write that checkpoint's absolute path as one plain-text line
to `ATARAXOS_NEXT_CHECKPOINT_FILE`.

`ATARAXOS_PARENT_CHECKPOINT` is the policy path used to start collection. Its
sibling optimizer state is resumed when present; every successful update writes
the next policy and AdamW state together before publishing the next-policy path.

Pipeline collection requires a positive temperature: actions must actually be
sampled from the probability distribution stored beside them. Temperature zero
remains available for collect-only performance/evaluation runs, but those
argmax trajectories are rejected as updater input by the default pipeline.

The included updater is the default in pipeline mode:

```bash
MODE=pipeline ROUNDS=10 \
  ./tools/run_ataraxos_cuda.sh
```

The updater defaults are deliberately conservative: two passes over one
policy-homogeneous round, trajectory-balanced terminal Monte Carlo advantages,
PPO clipping at 0.10, W/L/D cross entropy, a small entropy bonus, exact behavior
KL regularization, and gradient clipping. The values can be overridden with the
`ATARAXOS_*` environment variables used by `tools/update_ataraxos_once.sh`.

## Still deferred

- historical-opponent sampling;
- promotion/rejection gates for newly updated checkpoints.

Those are learning-policy decisions. The collector already supplies the exact
on-policy data needed to implement and test them.
