# Train B1/B2 handoff

Updated: 2026-10-30

## Repository and current state

- Active checkout: `/Users/macuser/bot2/metamon_tauros_stage1`
- Working branch: `tauros`
- Push target: `train_b1b2/main` at `git@github.com:bsmathers/train_b1b2.git`
- Last pushed commit at this handoff: `70399953` (`Fix AWR validation leakage and checkpoint alias`)
- The authoritative model-history report is
  [`docs/MODEL_TRAINING_HISTORY.md`](docs/MODEL_TRAINING_HISTORY.md).
- `testing4_work` is an older, separate Snorlax experiment. Do not treat it as
  this repository or copy its full training loop back wholesale.
- `tools/differential_native_adapters.py` now performs a seeded legacy/binary
  adapter differential. The 500-game reference run (seed 20261029, 32 lanes,
  one host) matched every native engine byte, legal-choice vector, action,
  result, and repeated legacy replay record: 258 p1 wins, 237 p2 wins, five
  ties, and no 500-decision timeouts. Binary transport was 12.24x faster at
  this boundary (76,239 vs 6,229 games/min). The binary-to-model observation
  decoder remains unimplemented; do not claim model-facing equivalence yet.

The checkout contains uncommitted Ataraxos learner completion work:

- `metamon/env/vectorized/vector_env.py` has behavior-metadata recording,
  exact decisive-game quotas, and bounded asynchronous replay writing changes;
- `metamon/rl/ataraxos_replay.py` defines and validates the on-policy replay
  contract;
- `metamon/rl/ataraxos.py` is the native collector and sequential round
  orchestrator;
- `metamon/rl/ataraxos_update.py` is the conservative reference updater.

The focused contract tests and a real local collect-update-reload smoke test now
pass. The intended architecture and remaining limitations are documented in the
Ataraxos section below.

## Current model: unambiguous identity

The current published model is the restored validation-best checkpoint from:

```text
/workspace/b1-data/checkpoints/stage1_b1_epoch51_selected_actor/
  ckpts/policy_weights/policy_actor_bc_final.pt
```

On another machine, preserve the same relative path under its `SAVE_DIR`.
The model is the selected-good-model actor refinement anchored to Epoch 51 of
the earlier full-strong-corpus actor run. It is not Epoch475, not the discarded
first B1 actor phase, not a weight average, and not an AWR result.

The default AWR launcher uses this exact file as both the live initialization
and immutable actor/value anchor. It does not use the latest chronological
numbered epoch unless `SOURCE_CHECKPOINT` is explicitly overridden.

## Direct model lineage

1. **TaurosV1 Epoch475 (`mediumg1online`)** supplied the initial actor and
   shared encoders. It originally had one actor and four shaped-return Q critic
   heads. The sharded artifact in `artifacts/epoch475/` reconstructs
   `policy_epoch_475.pt`.
2. **B1 value conversion** removed the four Q critics and their targets and
   introduced one categorical loss/draw/win head. The actor was initially
   frozen. The first value-only phase fitted only that head. The second phase
   allowed the encoders and value head to move under actor-distribution KL to
   Epoch475 and layer-weightable L2-SP on the backbone.
3. **Clean full-corpus actor BC** restarted from the accepted phase-2 value
   model, explicitly discarding the failed original actor phase. It trained the
   actor/backbone on all 311,696 local strong-model perspective trajectories,
   with actor KL and value-output anchoring. Epoch 51 from this run became the
   new anchor.
4. **Epoch-51 value refit** froze the Epoch-51 actor and both encoders and
   refitted only its W/L/D head on 70% full strong-model trajectories and 30%
   human trajectories.
5. **Epoch-51 selected actor refinement** loaded the value-refitted live model,
   kept the immutable actor anchor at Epoch 51, froze the refitted W/L/D head,
   and trained actor plus backbone on the 123,053-trajectory selected subset.
   Its restored validation-best `policy_actor_bc_final.pt` is the current model.

An older small Epoch20 policy was used only as a tournament comparator; it is
not an ancestor. The proposed B2 weight-interpolation branch was abandoned and
deleted. All accepted offline rounds used AdamW; Muon was considered but not
adopted.

The selected subset is a zero-copy symlink view and contains only perspective
trajectories from:

```text
Tauros-v1:          14,865
Tauros-v2:           9,363
SmallG1Online-v2:   60,335
SmallG1Online-v3:   38,490
Total:             123,053
```

Tauros-v3 is deliberately not included.

## Architecture now

The current model retains the Epoch475 grouped observation/action interface,
the grouped per-timestep encoder, the six-layer 512-wide trajectory transformer
with eight attention heads, and the residual masked actor. Its trajectory
attention uses fused FlashAttention on CUDA with causal window `(32, 0)`.

The old critic ensemble is deleted. The useful inference outputs are:

- a legal-action-masked actor distribution; and
- a three-class W/L/D value distribution, normally reduced to
  `P(win) - P(loss)` when a scalar value is required.

There are no four Q critic heads in native B1 checkpoints.

## Dataset and split facts

- Full local strong-model corpus: 311,696 `gen1ou` perspective trajectories.
- Initial B1 value mixture: 50% full strong-model, 40% human, 10% bounded
  `pac-tauros`.
- The `pac-tauros` provisioner streams the archive and retains a deterministic
  250,000-trajectory subset instead of expanding the full dataset.
- Epoch-51 value refit: 70% full strong-model and 30% human.
- Epoch-51 selected actor and AWR: 100% selected strong-model subset above.
- Dataset splitting now groups both player perspectives from one Showdown
  battle. Before commit `70399953`, 1,147 battle IDs crossed the selected
  train/validation boundary and 9.32% of validation trajectories had the other
  perspective in training. The already-trained selected actor predates that
  correction, so its imitation-validation checkpoint selection may be mildly
  optimistic. Its independent native tournaments remain valid evidence.

Never resume or recompute the separately frozen teacher-target corpus at
`replay_bc/cache/tauros_strict_v2_full_20260914` unless the user explicitly
reverses that instruction.

## Measurements worth retaining

The phase-2 W/L/D head was evaluated on 6,000 trajectories per source:

| Source | NLL | Accuracy | Brier | ECE |
|---|---:|---:|---:|---:|
| Human | 0.5258 | 72.676% | 0.3520 | 0.0298 |
| pac-tauros | 0.5014 | 74.322% | 0.3328 | 0.0200 |
| Strong-model | 0.4068 | 81.921% | 0.2577 | 0.0173 |

As expected, accuracy was weakest early in battles and strongest late. Draws
were nearly absent in these corpora, so draw calibration was not measured
meaningfully.

The first clean actor BC run showed held-out loss falling from `0.642587`
initially to `0.502743` after epoch 0 and `0.466675` by epoch 7. It continued
through Epoch 51, which became the later pipeline anchor.

The Epoch-51 value-only refit reached approximately `0.4309` validation loss.
It stopped because improvements became smaller than the configured `0.001`
minimum-delta for the patience window, not because the loss diverged.

A reported 1,250-game native test of the new selected model against TaurosV0
had win rate `0.594`. Preserve the underlying tournament logs with the model;
the statement alone is not enough to reproduce pairing, side balance, or seed.

## Weight averaging

The first weight-soup screen was invalid. It included a numbered checkpoint
that was actually a restored-best alias and averaged SigmaReparam `u`/`v`
spectral buffers directly. That produced implausible results, including a
last-16 score of `0.142`.

The corrected code removes a terminal restored-best alias and recalibrates all
24 SigmaReparam layers (48 `u`/`v` buffers) after averaging. Its 500-game screen
was:

```text
uniform epochs 14-29: 0.602
published best:       0.593
uniform epochs 26-29: 0.591
uniform epochs 28-29: 0.587
uniform epochs 22-29: 0.583
```

These are statistically similar at 500 games. No averaged candidate replaced
the current published best. If this is revisited, run at least 2,500 fresh,
paired games for confirmation and retain the generated manifest.

## Advantage-weighted cloning status

Offline actor training is now closed. Anchored AWR used the selected
123,053-trajectory subset—not the full strong corpus—and produced a 500-game
score of 0.570 against TaurosV0 (280 wins, 210 losses, 10 timeouts), statistically
indistinguishable from the non-AWR branch. An unanchored AWR experiment starting
from anchored-AWR epoch 30 scored 0.548 and was rejected. Do not use that
unanchored branch as a deployment model or as the source of future training.

For outcome `G in {-1, 0, 1}` and frozen source value
`V(s) = P(win|s) - P(loss|s)`:

```text
A = G - V(s)
w = clip(exp(A / 0.5), 0.25, 4.0)
loss = 0.55 normalized weighted NLL
     + 0.25 KL(source actor || live actor)
     + 0.20 KL(source value || live value)
```

The W/L/D head is frozen; actor and backbone train. The source checkpoint is
also the immutable actor/value reference. Numbered checkpoints are now strictly
chronological; the restored winner is written only as
`policy_actor_awr_final.pt` and `latest/policy.pt`.

## Native simulator and evaluation rules

- Gen 1 self-play/tournaments use the Zig `@pkmn/engine` addon through
  `metamon/env/vectorized/battle_host_engine.js`; they do not use a Showdown
  websocket server.
- Run `./tools/setup_pkmn_engine.sh` after a fresh checkout.
- `native_matchup.py` checks private request ownership, legal masks, committed
  actions, physical winner versus parsed POV outcome, side balance, and
  timeout/draw accounting.
- Argmax evaluation is requested with temperature zero, implemented as sampling
  disabled rather than dividing logits by zero.
- Tournaments must balance physical p1/p2 and use the same team distribution on
  both sides when comparing policies. Timeouts/draws count as half a point and
  must not be silently labelled losses.

## Ataraxos infrastructure direction

The next online stage should preserve the current actor plus W/L/D head and use
native self-play. Infrastructure requirements are:

1. immutable behavior-policy checkpoint per collection round;
2. exact per-decision behavior log probabilities, scalar W/L value, legal mask,
   committed action, terminal outcome, and policy identity;
3. one batched inference call per side/decision wave and one packed GPU-to-host
   transfer for learner-side action plus metadata;
4. persistent native lanes and KV state during a round;
5. bounded asynchronous replay compression so disk writing does not stall the
   simulator;
6. decisive-game quotas that reject unresolved time-limit games rather than
   labelling them losses;
7. a fully validated FIFO/round boundary before any update consumes it; and
8. a synchronous update hook that keeps the learner replaceable without
   changing collector plumbing.

The collector satisfies the requirements above. The included reference learner
uses terminal Monte Carlo advantages, trajectory-balanced clipped PPO,
categorical W/L/D cross entropy, exact behavior KL, entropy regularization, and
gradient clipping. AdamW state persists across rounds. Before any optimizer
step it replays complete recurrent
trajectories and rejects the round if the reconstructed actor distribution does
not match the stored behavior distribution.

A local native-Zig smoke loaded the selected actor, collected exactly two
decisive games (one win and one loss, 99 decisions), validated every stored row,
reconstructed the behavior policy at KL `0.00000006`, completed one optimizer
step, atomically published a 555-key checkpoint, and reloaded it strictly with
all floating tensors finite. `tools/run_ataraxos_cuda.sh` is the one-shot entry
point; `MODE=pipeline` now defaults to `tools/update_ataraxos_once.sh`.

The first performance pass caches native legality results across observation
and action translation, batches opponent feedback, and moves state conversion
plus encoding/compression to writer threads. The second pass adds semantics-
preserving lightweight Turn/Pokemon/Move history copies and a verified one-pass
DefaultActionSpace legality calculation. On the repeatable 32-lane adapter
benchmark, throughput increased from 618.5 to about 1,120 lane-steps/second
(about 81%). `PROFILE=1` enables production-host boundary timings; see
`docs/ATARAXOS_PERFORMANCE.md`.

The legacy text boundary was then isolated as the main simulator bottleneck.
Commit `9b91c800` adds an opt-in binary native-state transport: each decision
emits the Zig engine's authoritative 384-byte battle image, encoded result, and
legal engine choices for both sides in one frame. It completely skips JS log
decoding/request construction and Python Showdown/replay parsing. The included
lossless trajectory handler enforces state/action alignment. A 10,000-game,
64-lane local random-vs-random run reached about 158,000 games/minute and
274,000 lane-steps/second with one host, versus roughly 2,400 games/minute
through the full compatibility adapter. Four hosts were slower, so use one
native host per collector process at this boundary.

Binary state is not yet the model-facing default. The remaining integration is
an exact per-POV decoder into `GroupedObservationSpace`, including revealed-move
history and hidden-information rules, checked differentially against the text
adapter. Do not enable `METAMON_NATIVE_BINARY_STATE=1` in training until that
decoder is complete; the current binary mode is for raw trajectory collection
and boundary profiling.

The collector also defaults to one joint forward for both self-play sides while
keeping independent RL2/KV state. A deterministic native comparison matched all
8 outcomes and 542 decisions and improved local throughput from 78.2 to 105.2
games/minute. Set `JOINT_INFERENCE=0` for the separate-forward fallback.

Still deliberately unresolved:

- historical-opponent sampling and checkpoint promotion gates;
- performance profiling at production CUDA scale; and
- a destructive-write/failure-injection test for asynchronous replay output.

Do not import the old `testing4_work` update logic uncritically. It contains a
large amount of Snorlax-specific scheduling, retention, and bootstrap machinery
that is not part of the current model.

## Checkpoints and artifacts to back up

At minimum preserve:

```text
mediumg1online/ckpts/policy_weights/policy_epoch_475.pt
stage1_b1/ckpts/policy_weights/policy_value_backbone_best.pt
stage1_b1/ckpts/policy_weights/policy_phase2_final.pt
stage1_b1_actor_bc/ckpts/policy_weights/policy_epoch_51.pt
stage1_b1_epoch51_value_refit/ckpts/policy_weights/policy_value_head_refit_final.pt
stage1_b1_epoch51_selected_actor/ckpts/policy_weights/policy_actor_bc_final.pt
```

Also preserve the selected actor's chronological epochs 14-29 if weight soups
may be revisited, every `.manifest.txt` for an averaged checkpoint, tournament
game logs, run logs, flattened dataset configs, and the exact git commit.

## Immediate safe next steps

1. Run a small CUDA collect-update round (for example 16 lanes and 32 decisive
   games) and confirm initial behavior KL remains near zero with BF16.
2. Add a fixed-anchor evaluation/promotion gate before a long multi-round run.
3. Add historical-opponent sampling after the current-policy path is stable.
