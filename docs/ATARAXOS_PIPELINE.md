# Ataraxos native self-play infrastructure

This is the native collection, Ataraxos policy/value update, and sequential
round-orchestration layer for the next online stage.

## What is implemented

- Battles use the repository's native Zig `@pkmn/engine` adapter, not a
  Pokemon Showdown websocket server.
- Both players use batched transformer inference and independent persistent
  RL2/KV state. Self-play shares the read-only policy weights in GPU memory.
- Each recorded decision contains the committed action, exact behavior log
  probabilities, legal-action mask, complete categorical W/L/D distribution,
  and immutable behavior checkpoint identity—not only the scalar
  `P(win) - P(loss)` projection.
- Only terminal games enter an Ataraxos round. Natural simulator draws receive
  the draw target; time-limited or otherwise unfinished games are excluded
  rather than mislabeled as draws or losses.
- Replay compression runs in a bounded writer pool and is flushed before a
  round is validated.
- A completed round is rejected if probability mass, legality, action
  alignment, outcomes, or behavior-policy identity are inconsistent.
- Every completed physical game produces two independently validated player
  trajectories, one from each private POV. Opponent-only decisions such as
  forced switches retain their exact policy metadata.
- Each player's team is sampled independently per game from a 50/50 mixture of
  `b1_good_807` and the May-2026 high-ladder `hl_05_26` set. Mixture weights
  apply to source sets, not to their unequal numbers of files.
- Round summaries and progress logs are plain text.
- The learner replays complete recurrent trajectories, verifies the
  pre-update policy against the exact stored behavior probabilities, and then
  applies unnormalized GAE(0.5), categorical TD(0.8), clipped PPO(0.2), forward
  behavior KL, and the scheduled uniform-policy magnet.
- Each optimizer update weights all valid decisions equally over accumulated recurrent microbatches. There is one pass per rollout round, no advantage filtering or normalization, AdamW has zero weight decay, and gradients clip at 0.267.
- Learning-rate and magnet coefficients follow the original power schedules.
  Optimizer, schedule iteration, and parameter-only EMA(0.999) state persist
  atomically across rounds. Live SigmaReparam buffers are copied into EMA
  checkpoints instead of averaged.
- Checkpoints and round/update summaries are published atomically.
- The learner can run under `torchrun` DDP, compiles its tensor-only recurrent
  loss graph, buckets full histories by length to reduce padding, and reuses a
  persistent Inductor cache across update processes.
- A synchronous native evaluation manager gates checkpoint promotion every ten
  rounds by default. Its result and promotion decision are plain text.

## One collection round

On a provisioned CUDA host:

```bash
cd /workspace/train_b1b2
PYTHON_BIN=/workspace/b1-venv/bin/python \
METAMON_CACHE_DIR=/workspace/b1-data/cache \
SAVE_DIR=/workspace/b1-data/checkpoints \
CHECKPOINT=/workspace/b1-data/checkpoints/stage1_b1_epoch51_selected_actor/ckpts/policy_weights/policy_actor_bc_final.pt \
OUTPUT_DIR=/workspace/b1-data/ataraxos \
LANES=64 SIM_WORKERS=4 \
./tools/run_ataraxos_cuda.sh
```

The Python adapter is largely single-process bound. On CPU-rich GPU hosts,
`COLLECTOR_SHARDS` runs independent adapter/simulator processes against the
same immutable behavior checkpoint, then validates and merges their disjoint
replay shards before learning. `LANES` is per process. Start a 3090 profile at
`COLLECTOR_SHARDS=2 LANES=16 SIM_WORKERS=1`, then compare 1, 2 and 4 shards;
do not assume more processes help once CUDA inference becomes saturated.

By default each round targets 80,000 **valid decisions from completed games** and then drains all in-flight battles without starting replacements. With roughly 50 decisions per POV trajectory, this is approximately 800 games, 1,600 trajectories, and 200 optimizer updates at 400 decisions per update. Set `TARGET_DECISIONS` to adjust; `TARGET_DECISIONS=0 GAMES=128` restores the old exact-game quota. The actual decision count may exceed the target during draining and is logged per round.

`ATARAXOS_TARGET_BATCH_DECISIONS=400` sets the target valid decisions per optimizer step; `ATARAXOS_BATCH_SIZE=4` is the maximum number of recurrent trajectories in a GPU microbatch. Gradient accumulation weights each microbatch by its valid decisions, so short games do not get more weight. Initial replay validation now also checks W/L/D value distributions, with `ATARAXOS_MAX_INITIAL_VALUE_L1=0.05` default.

The round can be inspected without loading a model:

```bash
/workspace/b1-venv/bin/python -m metamon.rl.ataraxos \
  --mode inspect \
  --output-dir /workspace/b1-data/ataraxos \
  --round 0
```

Rerunning an interrupted round validates the existing files, verifies that
their behavior-policy hash matches, and collects only the missing terminal
games.

The replay contract is now collector version 2 because it stores categorical
values and explicit terminal outcomes. Version-1 rounds cannot supply the new
TD targets and are rejected; remove any old, not-yet-trained round directory
before starting this learner.

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
sibling `ataraxos_training_state.pt` is resumed when present; every successful
update writes the next policy, AdamW state, schedule iteration, and EMA together
before publishing the next-policy path.

Pipeline collection requires a positive temperature: actions must actually be
sampled from the probability distribution stored beside them. Temperature zero
remains available for collect-only performance/evaluation runs, but those
argmax trajectories are rejected as updater input by the default pipeline.

The included updater is the default in pipeline mode:

```bash
MODE=pipeline ROUNDS=10 \
  ./tools/run_ataraxos_cuda.sh
```

For a multi-GPU learner, set `ATARAXOS_GPUS` to the number of local CUDA
devices. Collection remains on `DEVICE` and the update hook launches one DDP
process per GPU:

```bash
MODE=pipeline ROUNDS=100 ATARAXOS_GPUS=4 EVALUATION_INTERVAL=10 \
  ./tools/run_ataraxos_cuda.sh
```

Compilation is enabled by default. Set `ATARAXOS_COMPILE=0` only for debugging.
The native promotion test defaults to 128 paired games per checkpoint against
TaurosV0. A candidate may trail its parent by at most two score points. Tune it
with `ATARAXOS_EVALUATION_GAMES`, `ATARAXOS_PROMOTION_SCORE_FLOOR`,
`ATARAXOS_PROMOTION_MAX_REGRESSION`,
`ATARAXOS_EVALUATION_LANES`, and `ATARAXOS_EVALUATION_WORKERS`.

The updater defaults reproduce the original Ataraxos optimization settings,
apart from deliberately retaining every decision: one pass, GAE lambda 0.5,
categorical TD lambda 0.8, PPO clipping at 0.20, policy/value coefficients 1.0,
forward behavior KL coefficient 0.10, zero weight decay, gradient clipping at
0.267, and EMA decay 0.999. The original learning-rate schedule begins at its
`1e-4` ceiling. That is aggressive for a pretrained Pokemon policy; use a small
collect-update-evaluate promotion test before a long run. Set `ATARAXOS_LR` to
override the schedule during such a safety comparison.

## Still deferred

- live-game streaming and truncated bootstrap targets;
- historical-opponent sampling;

Live-game streaming cannot safely retain transformer KV caches across a weight
update. It must first reconstruct each live history under the new policy. The
complete-game implementation avoids mixing hidden states from different policy
versions while using the same GAE/TD recurrences at terminal boundaries.

Those require separate collector or promotion-policy work. The current replay
still supplies exact on-policy data for complete-game updates.

## Remaining differences from upstream Ataraxos

- Upstream keeps games alive across updates after about 100 decisions per player and environment. This implementation instead finishes games and drains in-flight games at a completed-decision target, avoiding live-KV reconstruction and truncated bootstrap.
- The original filters to roughly the top quartile of absolute advantages. We
  intentionally retain all decisions, as requested, so actor/value/regularizer
  gradient balance will not be identical.
- Pokemon's recurrent transformer replays complete sequences; upstream Stratego batches single information states. Effective decision-count minibatches approximately match Ataraxos's post-filter 400 decisions, but their memory and computational costs differ.
- Upstream's evaluation manager uses its own Stratego population and rating
  regime. This implementation provides the same promotion-management boundary
  but defaults to a fixed native TaurosV0 safety gate.
- Stratego's learned starting-arrangement policy has no direct Pokemon analog.
  The fixed 50/50 team-source mixture is the explicit Pokemon replacement.
