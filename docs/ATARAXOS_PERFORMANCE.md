# Ataraxos collector performance

## Profiling method

The collector has two complementary profilers:

- `--profile` measures model construction, reset, evaluated-policy inference,
  environment steps, replay writing, and validation. It also enables exclusive
  timing sections inside the native adapter.
- Python `cProfile` was used locally to identify individual Python functions.
  Its absolute throughput is not representative because it substantially slows
  the parser and model, but before/after runs used the same workload.

For a model-independent completed-battle benchmark, use two policies that
sample uniformly from the live legal-action masks:

```bash
METAMON_CACHE_DIR=/workspace/b1-data/cache \
python tools/benchmark_native_random_selfplay.py \
  --games 500 --lanes 8 --workers 1 \
  --format gen1ou --team-set competitive --raw-team-files
```

This runs through the full Ataraxos-facing adapter: both private POV parsers,
legal masks, shaped reward/feedback state, automatic lane restarts, terminal
POV checks, and native timing frames are included. It does not include model
inference or replay compression, so it is the simulator/adapter ceiling rather
than production collector throughput.

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

## Legal-random completed-game benchmark

A dedicated deterministic Gen 1 OU run was added because fixed vector-step
rates do not account for game length or timeouts. Both players now sample only
legal actions; the old random opponent sampled across all 13 actions and relied
on repair, so it was not a true legal-random baseline.

The first 32-lane, two-worker, 250-game run completed at 616.5 games/minute.
Profiling identified two remaining avoidable conversion layers:

- replay `Move.__deepcopy__` used dynamic slot lookup millions of times per
  profiled run; its complete eight-slot copy is now direct assignment;
- native state construction converted replay Pokemon into poke-env Pokemon and
  immediately converted them again into universal observations. It now routes
  through the canonical replay-state converter directly. An opt-in verifier
  (`METAMON_NATIVE_VERIFY_DIRECT_STATE=1`) compared every field with the old
  route for 60 vector steps without a disagreement.

The direct route also exposed and fixed an independent correctness bug:
`UniversalPokemon.universal_types` padded a caller's live type list in place.
It now copies the list before adding the monotype placeholder. Zero/unknown
maximum HP is handled without division by zero in the replay converter.

The final profiled 500-game run on the local Mac used 8 lanes and one native
worker:

```text
completed games:              500
decisive games:               470
natural draws / timeouts:     3 / 27
completed games/minute:       922.7
decisive games/minute:        867.3
lane-steps/second:            1,552.5
```

The identical short workload improved from 616.5 to 690.6 profiled games/minute
(about 12%) at the original 32-lane setting. A no-profile lane/worker sweep
reached 944.7 games/minute at 8/1, 834.3 at 16/1, 727.0 at 32/1, 649.7 at 64/2,
and 637.6 at 64/4. The raw simulator prefers smaller lockstep groups because one
slow battle stalls the whole group. Neural inference may prefer a larger batch,
so tune production collection separately. This machine is Python-parser bound;
adding native workers inside one Python process cannot help.

In the final run, steady Zig `battle.update` consumed 0.020 seconds while the
native host's adaptation consumed 1.393 seconds. Across the Python pump
boundary, 93.8% was protocol dispatch/state parsing, 5.2% waiting for host
output, and 0.9% sending commands. The remaining optimization target is the
Python replay parser/state adapter; changing the Zig simulator is unlikely to
move end-to-end throughput materially.

### Historyless online adapter

The offline parser originally retained two kinds of snapshots that native
self-play never consumes: every completed turn and a deep-copied battle at each
mid-turn forced switch. Native lanes now opt into a historyless parser mode.
It advances the live turn in place, clears the same ephemeral fields as the
offline copy path, and uses a lightweight filled-subturn marker. Offline replay
parsing retains its original full-history behavior.

The online observation path also now:

- caches immutable cleaned move metadata while copying live PP into snapshots;
- caches Pokemon and move snapshots by all observation-visible mutable fields;
- derives the 13-way legal mask directly from the private simulator request;
- translates an action directly to a Showdown choice from the universal state
  and request, avoiding a replay-Pokemon -> poke-env-Pokemon -> BattleOrder
  round trip; and
- checks whether an active Pokemon has materialized without constructing a
  poke-env wrapper.

The old implementations remain available behind verification environment
variables. A 250-game run with direct-state, fast-legality, and fast-choice
verification enabled found no disagreement. Without verification, the final
deterministic 500-game run produced exactly the same 230-240 result, three
natural draws, 27 decision-limit timeouts, and 6,310 vector steps as the
pre-optimization run:

```text
completed games/minute:       2,434.7
decisive games/minute:        2,288.6
lane-steps/second:            4,096.8
```

That is 2.64x the original 922.7 games/minute full-adapter baseline. In this
run, steady Zig execution was still only 1.3% of native-host active time. The
largest remaining bucket is protocol emission and Python replay parsing; the
model-free reward/observation boundary is secondary.

## Feeding an Ataraxos learner

The recommended scaling boundary is a complete decisive trajectory, tagged by
an immutable behavior-policy hash. Keep each native environment and its two POV
parsers alive for an entire collection round. The joint policy driver batches
both physical sides in one GPU forward while retaining independent RL2/KV
state. Completed games enter a bounded producer queue; writer threads persist
the exact action, legal mask, behavior log probabilities, behavior value,
outcome, and policy id without blocking the simulator. The learner consumes
only a fully validated policy-homogeneous round.

For higher CPU throughput, use several independent 8-lane/one-worker Python
parser shards feeding one central GPU inference batcher and one bounded
trajectory queue. The central batcher recovers large GPU batches without
forcing unrelated battles into one lockstep parser group. Switch behavior
checkpoints only between games, drain the old policy version before an update,
and never mix policy hashes in a learner batch. This preserves on-policy ratios
while parallelizing the parser work that the profile shows is the real
bottleneck. This multi-process inference service is the next infrastructure
step; the current collector remains a single process with joint two-side
inference.

Full validation of a synthetic 2,000-replay round took 4.43 seconds
sequentially. Python threads made it slower: 2/4/8 workers took 5.00/5.46/5.93
seconds. Validation therefore remains sequential by default. It is a round
boundary cost, not a simulator-thread cost, and the strict full pass is retained
for correctness.
