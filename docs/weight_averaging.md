# Testing weight-averaged B1 policies

Weight averaging is treated as model selection, not as an automatic upgrade.
Only architecture-identical policy checkpoints may be averaged. Floating-point
parameters and buffers are averaged in FP32 and then restored to their original
dtype. Integer and Boolean state must match exactly.

## Screening

The screening launcher compares four candidates under identical conditions:

- the published, validation-best `policy_actor_bc_final.pt` control;
- uniform averages of the last 2, 4, and 8 numbered checkpoints.

Every candidate plays 500 games against TaurosV0. Both sides use the bundled
807-team good pool, candidate sides are balanced, action selection is argmax,
and every candidate uses the same seed. Results and checkpoint manifests are
written beneath the selected output directory. Existing completed sides are
validated and reused after an interrupted launch.

If early stopping left a long patience tail, set `END_EPOCH` to the actor epoch
with the best held-out imitation metric. This makes each window end at that
epoch rather than at the largest checkpoint number. Never infer the best epoch
from the checkpoint filename alone: the published final file is restored from
the validation best. Earlier numbered files record the training trajectory,
while the largest numbered file may also be overwritten with the restored best
when training publishes its final artifacts.

## Confirmation

Select at most the two best screening candidates. Re-test each, together with
the unaveraged control, for at least 2,500 games using a fresh seed and a new
output directory. The averaged model should be adopted only if it beats the
control in confirmation; a small win in the 500-game screen is not sufficient.
Keep the exact `.pt.manifest.txt` file with every averaged checkpoint.

The tournament summary is human-readable in `scores.txt`; machine-readable
ranking data is in `scores.csv`. Per-game progress is in each candidate's
`games.log`.
