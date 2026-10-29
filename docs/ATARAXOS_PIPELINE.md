# Ataraxos native self-play infrastructure

This is the collection and round-orchestration layer for the next online stage.
The policy/value update objective is deliberately not implemented yet. That
choice can change without changing or weakening the rollout contract.

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
ATARAXOS_NEXT_CHECKPOINT_FILE
```

`ATARAXOS_BEHAVIOR_CHECKPOINT` is the immutable content-addressed copy that
actually generated the round. The updater must atomically publish its new
checkpoint, then write that checkpoint's absolute path as one plain-text line
to `ATARAXOS_NEXT_CHECKPOINT_FILE`.

Example orchestration after an updater exists:

```bash
MODE=pipeline ROUNDS=10 UPDATE_COMMAND='/workspace/train_b1b2/tools/update_ataraxos_once.sh' \
  ./tools/run_ataraxos_cuda.sh
```

## Intentionally deferred

- the advantage estimator and policy/value loss;
- optimizer, clipping, KL control, and replay minibatch schedule;
- historical-opponent sampling;
- promotion/rejection gates for newly updated checkpoints.

Those are learning-policy decisions. The collector already supplies the exact
on-policy data needed to implement and test them.
