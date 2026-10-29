# Ataraxos collector performance

## Profiling method

The collector has two complementary profilers:

- `--profile` measures model construction, reset, evaluated-policy inference,
  environment steps, replay writing, and validation. It also enables exclusive
  timing sections inside the native adapter.
- Python `cProfile` was used locally to identify individual Python functions.
  Its absolute throughput is not representative because it substantially slows
  the parser and model, but before/after runs used the same workload.

The local comparison used `policy_epoch_51.pt`, four lanes, one native worker,
eight decisive games, argmax actions, and CPU inference.

## Findings and changes

Before legality caching, `legal_action_indices` consumed 1.49 seconds and
`definitely_valid_actions` consumed 1.48 seconds. The same legal set was built
once for the observation and immediately rebuilt to translate the chosen
action. Caching by lane, side, parsed-state mutation serial, and action space
reduced those figures to 0.72 and 0.72 seconds. Total profiled collection time
fell from 12.14 to 10.46 seconds, about 14%, and profiled games/minute rose from
42.4 to 49.6.

The next dominant Python parser function was
`Turn.create_next_turn -> copy.deepcopy`. The optimized path retains immutable
dex data, copies all battle-mutating Pokemon state, preserves cross-object
aliases through the deepcopy memo, and copies poke-env Move slots directly
instead of reconstructing every move through a dex lookup. This keeps complete
turn-history semantics while removing most generic-copy dispatch.

The native collector always uses the fixed 13-way `DefaultActionSpace`.
Legality is now computed with the same move/switch membership rules in one pass
instead of constructing a temporary poke-env `BattleOrder` for every candidate.
An opt-in verifier (`METAMON_NATIVE_VERIFY_FAST_LEGAL=1`) compares the fast and
generic masks on every decision. A deterministic 32-lane, 120-step Gen 1 run
completed without a disagreement.

On the repeatable 32-lane adapter benchmark, these parser and legality changes
raised throughput from 618.5 to 1,119.7 lane-steps/second (about 81%). The
intermediate state-copy-only result was 855.3 lane-steps/second. These figures
exclude neural inference and should be read as evidence that the CPU adapter
headroom improved, not as a prediction of end-to-end CUDA games/minute.

Two smaller changes remove work from the simulator thread:

- opponent RL2 feedback is applied as one vectorized batch instead of one
  Python call per active lane;
- `UniversalState.to_dict`, JSON encoding, and LZ4 compression all run in the
  bounded replay-writer pool. The final clean run spent effectively zero time
  waiting for writer flush.

The clean post-change timing for 150 vector steps was:

```text
evaluated policy:             1.870 s
environment step:             3.556 s
  opponent policy:            2.016 s
  native settle + advance:    1.014 s
  observations/rewards/etc:   0.526 s
environment reset:            0.665 s
environment construction:     0.406 s
model construction/load:      0.156 s
replay validation:             0.018 s
writer flush:                  0.000 s
```

This run reached 78.2 games/minute locally. That number is only a regression
benchmark, not a CUDA throughput prediction.

The two physical players originally used separate policy calls. A joint driver
now concatenates both player batches, retains separate RL2 and KV-cache slices,
and returns evaluated actions/metadata plus opponent actions in one packed
device-to-host transfer. Opponent-only force-switch waves use a view of the
opponent KV slice and therefore remain sequentially correct.

On the same deterministic workload, joint inference produced exactly the same
8 games, 5-3 outcome, and 542 recorded decisions. Throughput rose from 78.2 to
105.2 games/minute (about 35%). Total standard-wave policy time changed from
3.89 seconds across two calls to a 2.34-second combined boundary in one call
(the latter measurement also included construction of opponent observations;
new profiles report that separately). Joint inference is now the default; set
`JOINT_INFERENCE=0` to use the older independent forwards.

## Profile on the target CUDA host

Run a short representative round with the production lane/worker settings:

```bash
cd /workspace/train_b1b2
PYTHON_BIN=/workspace/b1-venv/bin/python \
METAMON_CACHE_DIR=/workspace/b1-data/cache \
SAVE_DIR=/workspace/b1-data/checkpoints \
OUTPUT_DIR=/workspace/b1-data/ataraxos-profile \
GAMES=256 LANES=64 SIM_WORKERS=4 WRITER_WORKERS=2 PROFILE=1 \
./tools/run_ataraxos_cuda.sh
```

Interpret the output as follows:

- If `evaluated_policy` still dominates with joint inference enabled, compare
  `JOINT_INFERENCE=0` on the target GPU. Large CUDA batches can scale differently
  from the local CPU benchmark, so retain whichever path wins there.
- If `pump_settle + advance_lanes` dominates, increase native workers only
  until CPU quota pressure appears; these sections include both Zig execution
  and protocol parsing.
- If `build_obs + rewards_restart + opponent_feedback` dominates, the Python
  adapter is the limit. Do not add simulator workers.
- If reset/construction is material relative to the round, keep the native
  process and team cache alive across update rounds. This matters most for
  small collection quotas.
- If `writer_flush` is nonzero, add one writer worker at a time. A zero flush
  means more writers cannot improve throughput.

The present local profile does not justify increasing writer workers above two.
Joint inference is already the default, but the target CUDA profile remains the
authority for lane and native-worker tuning.

The native boundary counters also make clear where further simulator-side work
would be misplaced. In the deterministic adapter profile, steady Zig
`battle.update` occupied about 1.4% of active host time; Node protocol adaptation
occupied 98.6%. After the parser changes, the Python pump boundary was 89.2%
protocol dispatch/state parsing, 10.4% waiting for native output, and 0.4%
command sends. The
collector was therefore parser/adapter-bound, not Zig-simulation-bound.

Full validation of a synthetic 2,000-replay round took 4.43 seconds
sequentially. Python threads made it slower: 2/4/8 workers took 5.00/5.46/5.93
seconds. Validation therefore remains sequential by default. It is a round
boundary cost, not a simulator-thread cost, and the strict full pass is retained
for correctness.
