"""Generate and index B2 interpolations for tournament selection.

Reuses stage1_interpolate.interpolate, which fixes SigmaReparam singular vectors
and synchronizes target networks after weight blending.
"""

import json
from pathlib import Path


def generate_interpolations(
    combinations, taurosv1_dir, save_dir, b1_run, b2_run, dry_run=False
):
    if not combinations:
        raise ValueError("No interpolation candidates")
    weights = save_dir / b2_run / "ckpts" / "policy_weights"
    rows = []
    for idx, (base, b1, alpha) in enumerate(combinations, 1):
        theta0 = (
            taurosv1_dir
            / "mediumg1online"
            / "ckpts"
            / "policy_weights"
            / f"policy_epoch_{base}.pt"
        )
        theta1 = (
            save_dir / b1_run / "ckpts" / "policy_weights" / f"policy_epoch_{b1}.pt"
        )
        if not theta0.is_file() or not theta1.is_file():
            raise FileNotFoundError(f"Missing base/B1 checkpoint: {theta0} or {theta1}")
        rows.append(
            {
                "id": idx,
                "base_epoch": base,
                "b1_epoch": b1,
                "alpha": alpha,
                "theta0": str(theta0.resolve()),
                "theta1": str(theta1.resolve()),
                "weights": str((weights / f"policy_epoch_{idx}.pt").resolve()),
            }
        )
    if dry_run:
        for row in rows:
            print(
                f"Would interpolate {row['theta0']} + {row['theta1']} "
                f"alpha={row['alpha']} -> {row['weights']}"
            )
        return rows
    weights.mkdir(parents=True, exist_ok=True)
    manifest_path = weights / "interpolation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != rows:
            raise ValueError(
                f"Existing B2 checkpoint IDs have different meanings: {manifest_path}. "
                "Use a new --b2-run-name."
            )
    else:
        if list(weights.glob("policy_epoch_*.pt")):
            raise ValueError(f"Existing B2 weights without manifest: {weights}")
        manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    import torch
    from metamon.rl.stage1_interpolate import interpolate

    cached_base = (None, None)
    cached_b1 = (None, None)
    for row in rows:
        destination = Path(row["weights"])
        if destination.is_file():
            continue
        if cached_base[0] != row["theta0"]:
            cached_base = (
                row["theta0"],
                torch.load(row["theta0"], map_location="cpu", weights_only=True),
            )
        if cached_b1[0] != row["theta1"]:
            cached_b1 = (
                row["theta1"],
                torch.load(row["theta1"], map_location="cpu", weights_only=True),
            )
        if row["alpha"] == 0.0:
            state = cached_base[1]  # preserve exact TaurosV1 endpoint
        elif row["alpha"] == 1.0:
            state = cached_b1[1]  # preserve exact B1 endpoint
        else:
            state = interpolate(cached_base[1], cached_b1[1], row["alpha"])
        partial = destination.with_suffix(".pt.tmp")
        torch.save(state, partial)
        partial.replace(destination)
        print(
            f"Generated B2 #{row['id']}: base={row['base_epoch']} "
            f"B1={row['b1_epoch']} alpha={row['alpha']}"
        )
    return rows
