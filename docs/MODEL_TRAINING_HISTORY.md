# Current model: complete training and fine-tuning history

## Executive summary

The current policy is a Gen 1 OU actor plus categorical W/L/D value model. It
descends from TaurosV1 Epoch475, but several experimental branches were rejected
before the current lineage was established.

The accepted path was:

```text
TaurosV1 Epoch475
  -> delete four Q critics; add W/L/D head
  -> value-head-only fitting
  -> constrained value + backbone fitting
  -> full-strong-corpus actor behavior cloning
  -> choose chronological Epoch 51 as a new anchor
  -> value-head-only refit anchored to Epoch 51
  -> selected-strong-policy actor/backbone refinement
  -> restored validation-best selected actor (current model)
```

The current checkpoint is normally:

```text
/workspace/b1-data/checkpoints/stage1_b1_epoch51_selected_actor/
  ckpts/policy_weights/policy_actor_bc_final.pt
```

The corrected AWR code is a proposed continuation of this checkpoint, not part
of the current model. Corrected weight averages were screened but not adopted.

## 1. Starting point: TaurosV1 Epoch475

The original starting policy was `mediumg1online` Epoch475. It used the grouped
Metamon observation stack, a transformer policy, one masked actor head, and four
Q critic heads trained against shaped returns. The repository stores the large
checkpoint in shards under `artifacts/epoch475/`; `tools/assemble_epoch475.py`
reconstructs the ordinary AMAGO policy file.

Epoch475 was already a competent behavioral policy. The central difficulty was
not producing plausible actions; it was obtaining a value signal suitable for
online policy optimization without destroying the actor's broad competence.
The old four critics were tied to their historical shaped-return objective and
were not a clean win-probability estimate.

## 2. Abandoned early online/Snorlax experiments

Before this repository's final B1 design, the adjacent `testing4_work` checkout
was used for SnorlaxV1 online experiments. That work exposed important failure
modes:

- validation and collection code had several singleton-dimension mistakes;
- early offline/online updates could rapidly move a strong actor and collapse
  head-to-head performance despite decreasing training loss;
- simulator outcomes and time limits must be distinguished explicitly;
- replay retention, nominal opponent offsets, and rolling-checkpoint identity
  need exact contracts;
- fast simulation is irrelevant if policy transfer, model reload, or replay
  serialization dominates the boundary; and
- independent native tournaments are indispensable because imitation or PPO
  losses are not a direct gameplay metric.

Snorlax weights are not ancestors of the current B1 model. The lessons shaped
the stricter native adapter and the conservative offline pipeline here.

Two other reference branches were also not ancestors. An older small Epoch20
model, fine-tuned on Tauros-v1/Tauros-v2 scraped games, was retained only as a
head-to-head benchmark. The early B2 idea—interpolating B1 weights back toward
the base model to recover out-of-distribution strength—was abandoned and its
implementation deleted. It could not guarantee preservation of the specific
in-sample behavior learned by B1 and did not solve critic calibration.

## 3. Initial Stage-1 concept and why it changed

The first Stage-1 patch attempted conventional Metamon fine-tuning with the old
critic layout. That was rejected because offline imitation does not supply a
well-defined target for four shaped-return Q heads, and allowing shared features
to move could leave their outputs arbitrarily miscalibrated.

The model was redesigned around a single three-class value head:

```text
p_value(s) = softmax(W h(s) + b)
classes = [loss, draw, win]
V(s) = p_value(win|s) - p_value(loss|s)
```

The four critics, target critics, maximized critics, target actor, and PopArt
state were deleted from native B1 checkpoints. This made the value semantics
explicit and made later self-play infrastructure possible.

## 4. B1 phase 1: value-head-only training

Every state in a replay was labelled with the final result from that player's
perspective. Only the new W/L/D head trained; actor and both encoders were frozen
and held in evaluation mode so dropout and SigmaReparam buffers could not drift.

The phase used up to 100 epochs of 1,000 AdamW optimizer updates, with patience
3. The initial data mixture was:

- 50% full strong-model trajectories;
- 40% human Gen 1 OU trajectories; and
- 10% `pac-tauros` trajectories.

For disk-bounded machines, `pac-tauros` was streamed into a deterministic
250,000-trajectory reservoir rather than fully decompressed.

This phase answered a narrow question: how accurately can a linear categorical
head recover game outcome from the unchanged Epoch475 representation?

## 5. B1 phase 2: value plus constrained backbone

The actor remained frozen, while the W/L/D head and full shared backbone were
unfrozen. The phase minimized categorical value cross-entropy plus two
constraints:

```text
L_phase2 = CE_WLD
         + lambda_policy * KL(pi_Epoch475 || pi_live)
         + lambda_L2SP * mean_layer_weighted((theta - theta_Epoch475)^2)
```

The KL constrained functional action drift even though actor-head parameters
were frozen: changing the shared representation changes actor outputs. L2-SP
provided a separate parameter-space restraint, with longest-prefix layer
weights so selected encoder blocks could be protected more strongly.

This phase allowed up to 1,000 epochs, again with 1,000 optimizer updates per
epoch and patience 5. The best policy, optimizer, and scheduler were saved
atomically and restored on early stopping.

### Value evaluation

The resulting value head was evaluated on 6,000 sampled trajectories from each
source:

| Source | NLL | Frequency baseline | Accuracy | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Human | 0.5258 | 0.6930 | 72.676% | 0.3520 | 0.0298 |
| pac-tauros | 0.5014 | 0.6946 | 74.322% | 0.3328 | 0.0200 |
| Strong-model | 0.4068 | 0.5799 | 81.921% | 0.2577 | 0.0173 |

Accuracy increased from early to late battle states. Draw examples were almost
absent, so no strong conclusion about the draw class is justified.

## 6. Rejected original actor phase

The original B1 design immediately followed phase 2 with an advantage-weighted
actor/backbone phase. It stopped after a short patience window and performed
worse than expected in gameplay. A representative two-side test reported
`48-77` from one physical side and `81-44` from the other, or `129-121`
aggregated, which was materially below the expected strength of the starting
policy despite being close to 50% in that particular sample.

This branch was rejected. Its actor weights are not in the current lineage.
The value-trained phase-2 model was retained and the actor experiment was
restarted cleanly.

## 7. Clean full-strong-corpus actor behavior cloning

The replacement actor run started from the accepted phase-2 model and trained
on all 311,696 local strong-model perspective trajectories. The value head was
frozen. Actor and shared backbone trained with ordinary logged-action
cross-entropy plus functional anchors to the phase-2 source policy and value
model. This deliberately resembled the older Epoch20-style successful
fine-tuning recipe more than the rejected offline advantage objective.

The configured effective batch was 48: batch 12 with gradient accumulation 4.
Each epoch contained 1,000 optimizer updates and therefore displayed 4,000
microbatches in AMAGO's progress bar.

All accepted offline rounds used AdamW. Muon was considered but not adopted:
the run already had stable AdamW behavior and the main uncertainty was objective
and model selection, not optimizer throughput.

The held-out metric progressed as follows at the beginning:

| Candidate | Validation loss |
|---|---:|
| Initial | 0.642587 |
| Epoch 0 | 0.502743 |
| Epoch 1 | 0.489390 |
| Epoch 2 | 0.482370 |
| Epoch 3 | 0.477553 |
| Epoch 4 | 0.473920 |
| Epoch 5 | 0.471187 |
| Epoch 6 | 0.468494 |
| Epoch 7 | 0.466675 |

The run continued substantially farther. Chronological Epoch 51 was chosen as
the new anchor for all subsequent work. From that point forward, Epoch475 was
used only by the loader to recover architecture/observation metadata; its
weights and outputs were no longer fine-tuning anchors.

## 8. Epoch-51-anchored value refit

The next pipeline loaded the exact Epoch-51 checkpoint from the clean actor run
and froze actor plus both encoders. It refitted only the categorical W/L/D head
on:

- 70% full strong-model corpus; and
- 30% human replay corpus.

The best observed validation metric was around `0.4309`. The run stopped after
several improvements smaller than the configured `0.001` minimum delta. This
was expected early-stopping behavior, not an exception and not a rising loss.

The published output was:

```text
stage1_b1_epoch51_value_refit/ckpts/policy_weights/
  policy_value_head_refit_final.pt
```

## 9. Selected-good-model actor refinement

The final accepted offline round used:

- immutable actor anchor: clean actor Epoch 51;
- live initialization and frozen value reference: Epoch-51 value-refit final;
- trainable modules: actor plus shared backbone;
- frozen module: refitted W/L/D head; and
- training data: only selected strong-policy trajectories.

The selected replay view contains:

| Perspective player | Trajectories |
|---|---:|
| Tauros-v1 | 14,865 |
| Tauros-v2 | 9,363 |
| SmallG1Online-v2 | 60,335 |
| SmallG1Online-v3 | 38,490 |
| **Total** | **123,053** |

The view is assembled with symlinks, so it consumes negligible extra replay
storage. Tauros-v3 was explicitly replaced by Tauros-v1 in the final selector.

The restored validation winner was published as
`policy_actor_bc_final.pt`. A reported 1,250-game native test against TaurosV0
gave win rate `0.594`, a substantial improvement over TaurosV0. Tournament logs
remain the authoritative evidence because validation loss alone does not rank
game-playing policies reliably.

### Known validation caveat

This run occurred before the battle-grouped split correction. Individual
perspective filenames were shuffled independently, so two sides of the same
battle could cross the split. A later audit measured:

- 123,053 selected trajectories;
- 116,638 unique battles;
- 6,415 battles with both selected perspectives;
- 1,147 battle IDs crossing train and validation; and
- 9.32% of validation trajectories with the counterpart in training.

This can make the imitation-validation metric optimistic and can affect which
epoch early stopping selects. It does not contaminate independent native
tournaments against TaurosV0. Commit `70399953` fixes all future dataset splits
by grouping on Showdown battle ID.

## 10. Weight averaging experiments

### Invalid first screen

The initial soup script made two correctness mistakes:

1. it treated a restored-best file named like the terminal epoch as a distinct
   chronological checkpoint; and
2. it averaged SigmaReparam `u` and `v` spectral-estimate buffers directly.

That produced erratic scores, including `0.142` for a last-16 average. Those
results must not be used.

### Corrected screen

The corrected implementation detects/removes the restored-best alias and
recomputes every SigmaReparam buffer from the averaged weight with power
iteration. A 500-game screen produced:

| Candidate | Score vs TaurosV0 |
|---|---:|
| Uniform epochs 14-29 | 0.602 |
| Published validation best | 0.593 |
| Uniform epochs 26-29 | 0.591 |
| Uniform epochs 28-29 | 0.587 |
| Uniform epochs 22-29 | 0.583 |

At this sample size these candidates are approximately tied. The published
validation-best model remained the conservative default. Any adoption decision
requires at least 2,500 paired games with a fresh seed.

## 11. Corrected advantage-weighted cloning experiment

AWR is implemented as a new experiment starting from the current selected
actor final. It uses only the 123,053-trajectory selected subset.

For terminal outcome `G` and frozen current-model value `V(s)`:

```text
A(s,a_logged) = G - V(s)
w = clip(exp(A / 0.5), 0.25, 4.0)
```

The complete objective is:

```text
0.55 * weighted_action_NLL / mean(w)
+ 0.25 * KL(frozen_source_actor || live_actor)
+ 0.20 * KL(frozen_source_value || live_value)
```

The W/L/D head remains frozen; actor and backbone train. The same source
checkpoint initializes the live model and both functional anchors, so Epoch475
weights or predictions do not enter this objective.

The implementation was audited for tensor alignment, detach/gradient behavior,
legal and padding masks, and value/actor anchors. The split leakage described
above was fixed, and restored-best outputs no longer masquerade as numbered
chronological epochs. Anchored AWR later scored 0.570 over 500 games against
TaurosV0, with no statistically meaningful improvement over the non-AWR model.
An unanchored continuation from AWR epoch 30 scored 0.548 and was rejected.
Offline actor training was therefore closed; further actor improvement should
come from controlled native self-play.

## 12. Current inference model and future self-play

After removal of historical critic machinery, the model exposes exactly what is
needed for an Ataraxos-style online stage:

```text
pi(a | history)          legal-action-masked actor distribution
p(loss, draw, win | h)   categorical outcome head
V(h) = p(win|h) - p(loss|h)
```

The planned online infrastructure uses native Zig `@pkmn/engine` self-play,
batched policy inference, persistent KV caches, immutable behavior-policy
snapshots, and exact stored behavior metadata. The collector/update boundary
must be solved before choosing the final PPO/Ataraxos loss. In particular, an
update must never infer old log probabilities from a newer policy or treat a
time-limit boundary as a loss.

## 13. Reproducibility and preservation checklist

Preserve these artifacts together:

1. reconstructed Epoch475;
2. B1 phase-2 best/final;
3. clean actor Epoch 51;
4. Epoch-51 value-refit final;
5. selected actor `policy_actor_bc_final.pt`;
6. selected actor chronological epochs 14-29 if averaging may be revisited;
7. every weight-average manifest and candidate actually evaluated;
8. per-game native tournament logs and summaries;
9. run logs and W&B exports;
10. each run's flattened dataset config; and
11. git commit `70399953` or a later explicitly recorded successor.

The actor/value checkpoint alone is usable, but without its data selection,
anchors, split behavior, and tournament logs it is not scientifically
reproducible.
