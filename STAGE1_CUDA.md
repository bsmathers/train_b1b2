# Stage-1 B1/B2 CUDA training

The repository contains TaurosV1 Epoch475 as verified shards under
`artifacts/epoch475/`. The B1 launcher reconstructs the checkpoint automatically
under `checkpoints/` on its first run.

Requirements:

- A CUDA GPU with at least 24 GB VRAM; RTX 3090 is the cost-oriented default.
- CUDA-enabled PyTorch with BF16 support.
- Enough disk for the downloaded and extracted strong-model trajectory corpus.

Launch B1:

```bash
HF_TOKEN=optional-hugging-face-token \
WANDB_API_KEY=optional-key \
./tools/train_stage1_b1_cuda.sh
```

On first launch the script downloads the pinned `gen1ou.zip` archive from
`bsmathers/scraped_replays_parsed`, verifies its SHA-256, and safely extracts it
under `cache/strong_model_replays_parsed`. Later launches reuse that directory.
`HF_TOKEN` is optional for this public dataset. Set `GOOD_REPLAYS_DIR` to reuse
an existing local copy instead.

Defaults are batch size 12, one accumulation step, 1,000 updates per epoch,
30 epochs, a deterministic 10% held-out split, and actor-NLL early stopping
with patience 5. Override them with `BATCH_SIZE`, `GRAD_ACCUM`,
`STEPS_PER_EPOCH`, or `EPOCHS`.

The launcher defaults to eager execution because the installed AMAGO/PyTorch
combination has shown TorchDynamo failures. Standard PyTorch SDPA still uses
fused CUDA attention kernels. Set `ENABLE_TORCH_COMPILE=1` only after a short
machine-specific smoke test.


## Stage-1 checkpoint selection: balanced B1 and B2 tournaments

After B1 training, choose B1 epochs and B2 interpolation coefficients by head-to-head
matches against **TaurosV0 checkpoint 62**. The `good` pool contains the 807
Showdown teams supplied for this experiment; it is included reproducibly under
`artifacts/good_teams_807.bz2.base64`. The selector verifies its SHA-256 and
unpacks it into `cache/teams/b1_good_807/gen1ou` on first use. The `bad`
pool is the existing Hugging Face-backed `hl_05_26` team set.

For each candidate, the tournament tests all four **B1-or-B2 / TaurosV0**
assignments: good/good, good/bad, bad/good, bad/bad. Each condition is tested
in *both challenger/acceptor roles* for side-bias control. The score is the
equal-weighted average of these four win rates; draws count as half a win.
The per-condition scores are retained, and incomplete matches cannot select a
winner.

The existing local Showdown service must be running, and both participants
must be able to connect to it. Tournament matches use the normal H2H runner.

```bash
# The same checkpoint tree used by the B1 training launcher.
export METAMON_CACHE_DIR="$PWD/cache"
export TAUROSV1_SAVE_DIR="$PWD/checkpoints"

# Preview all B1 matches (no GPU work).
python tools/select_stage1_b1_epoch.py --mode b1 --dry-run

# Complete B1 checkpoint sweep, then test interpolations of its top 3 epochs
# against TaurosV1 Epoch 475 over five coefficients.
python tools/select_stage1_b1_epoch.py \
  --mode both \
  --save-dir "$PWD/checkpoints" \
  --gpus 0 \
  --battles-per-side 50 \
  --top-k 3 \
  --base-epochs 475 \
  --alphas 0.2 0.4 0.6 0.8 1.0
```

`--epochs` restricts the B1 checkpoint sweep; by default every saved B1
`policy_epoch_N.pt` is included. `--base-epochs` can include several
**compatible TaurosV1** base checkpoints. `--good-teams-zip` optionally
uses the original archive instead of the bundled data, but the checked-in
archive is sufficient on a fresh machine.

Output files under `checkpoints/stage1_selection`:

- `b1/candidate_scores.csv` and `b1/selected_epoch.json`
- `b2/candidate_scores.csv` and `b2/selected_interpolation.json`
- `b1/` and `b2/` raw per-match results, resuming only completed matchups
- `checkpoints/stage1_b2_selection/ckpts/policy_weights/` generated B2
  checkpoints, with `interpolation_manifest.json` mapping numeric IDs to
  base epoch, B1 epoch, and alpha

B2 uses `metamon.rl.stage1_interpolate.interpolate`, including its SigmaReparam
singular-vector handling and target-network synchronization. The candidate ID
is a numeric checkpoint index, **not** a percent-alpha value.

To run B2 separately after a completed B1 sweep, use `--mode b2` and the
same `--output-dir`. Changing the matchup grid, replay collection, or number
of battles requires a new output directory / B2 run name to avoid reusing
stale evaluation results. The script selects checkpoint files, but does not
begin further online or offline training.
