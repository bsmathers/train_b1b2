#!/usr/bin/env python3
"""Build and test uniform soups from a completed B1 actor run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.average_policy_checkpoints import average_checkpoints


EPOCH_RE = re.compile(r"policy_epoch_(\d+)\.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbered_checkpoints(policy_dir: Path, end_epoch: int | None) -> list[tuple[int, Path]]:
    found = []
    for path in policy_dir.glob("policy_epoch_*.pt"):
        match = EPOCH_RE.fullmatch(path.name)
        if match:
            epoch = int(match.group(1))
            if end_epoch is None or epoch <= end_epoch:
                found.append((epoch, path.resolve()))
    return sorted(found)


def _same_policy_state(first: Path, second: Path) -> bool:
    first_state = torch.load(first, map_location="cpu", weights_only=True)
    second_state = torch.load(second, map_location="cpu", weights_only=True)
    if list(first_state) != list(second_state):
        return False
    return all(
        first_state[key].shape == second_state[key].shape
        and first_state[key].dtype == second_state[key].dtype
        and torch.equal(first_state[key], second_state[key])
        for key in first_state
    )


def _remove_restored_best_alias(
    numbered: list[tuple[int, Path]], published_best: Path
) -> tuple[list[tuple[int, Path]], int | None]:
    """Drop the synthetic terminal epoch written after restoring validation-best."""
    if not numbered:
        return numbered, None
    terminal_epoch, terminal_path = numbered[-1]
    if _same_policy_state(terminal_path, published_best):
        return numbered[:-1], terminal_epoch
    return numbered, None


def _read_summary(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    labels = {
        "B1 wins": ("wins", int),
        "TaurosV0 wins": ("losses", int),
        "Natural draws": ("natural_draws", int),
        "Decision-limit timeouts": ("timeouts", int),
        "B1 match score with half points for draws/timeouts": ("score", float),
    }
    for line in path.read_text().splitlines():
        for label, (key, converter) in labels.items():
            if line.startswith(label + ":"):
                values[key] = converter(line.split(":", 1)[1].strip())
    if set(values) != {key for key, _ in labels.values()}:
        raise ValueError(f"incomplete tournament summary: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="stage1_b1_epoch51_selected_actor")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--sigma-power-iterations", type=int, default=32)
    parser.add_argument("--end-epoch", type=int)
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if min(args.battles, args.lanes, args.workers, args.timeout) < 1:
        parser.error("battles, lanes, workers, and timeout must be positive")
    if not args.windows or any(window < 2 for window in args.windows):
        parser.error("every averaging window must be at least two")

    save_dir = args.save_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    policy_dir = save_dir / args.run_name / "ckpts" / "policy_weights"
    published_best = policy_dir / "policy_actor_bc_final.pt"
    if not published_best.is_file():
        parser.error(f"published validation-best checkpoint is missing: {published_best}")
    numbered = _numbered_checkpoints(policy_dir, None)
    if not numbered:
        parser.error(f"no numbered checkpoints found in {policy_dir}")
    numbered, removed_alias = _remove_restored_best_alias(numbered, published_best)
    if removed_alias is not None:
        print(
            f"Excluded policy_epoch_{removed_alias}.pt: it is the restored validation-best "
            "alias, not a chronological training checkpoint",
            flush=True,
        )
    if args.end_epoch is not None:
        numbered = [(epoch, path) for epoch, path in numbered if epoch <= args.end_epoch]
    if not numbered:
        parser.error("no chronological numbered checkpoints remain after filtering")

    windows = sorted(set(args.windows))
    skipped_windows = [window for window in windows if window > len(numbered)]
    windows = [window for window in windows if window <= len(numbered)]
    if skipped_windows:
        print(
            f"Skipping averaging windows {skipped_windows}; only "
            f"{len(numbered)} numbered checkpoints exist",
            flush=True,
        )
    if not windows:
        parser.error("at least two numbered checkpoints are required for averaging")
    output_dir.mkdir(parents=True, exist_ok=True)
    averages_dir = output_dir / "averaged_checkpoints"
    candidates = [("published_best", published_best.resolve(), [])]
    for window in windows:
        selected = numbered[-window:]
        epochs = [epoch for epoch, _ in selected]
        name = f"uniform_last_{window}_epochs_{epochs[0]}_{epochs[-1]}"
        destination = averages_dir / f"{name}.pt"
        average_checkpoints(
            [path for _, path in selected],
            [1.0] * window,
            destination,
            sigma_power_iterations=args.sigma_power_iterations,
        )
        candidates.append((name, destination, epochs))

    rows = []
    for index, (name, checkpoint, epochs) in enumerate(candidates):
        tournament_dir = output_dir / "tournaments" / name
        snapshot = tournament_dir / "candidate_checkpoint.pt"
        if snapshot.is_file() and _sha256(snapshot) != _sha256(checkpoint):
            raise RuntimeError(
                f"candidate changed since this tournament began: {name}; "
                "choose a new output directory"
            )
        command = [
            args.python,
            str(ROOT / "tools" / "run_b1_vs_taurosv0_500.py"),
            "--checkpoint-path",
            str(checkpoint),
            "--battles",
            str(args.battles),
            "--candidate-team",
            "b1_good_807",
            "--opponent-team",
            "b1_good_807",
            "--save-dir",
            str(save_dir),
            "--run-name",
            args.run_name,
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(tournament_dir),
            "--lanes",
            str(args.lanes),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--temperature",
            "0",
            "--timeout",
            str(args.timeout),
        ]
        print(f"Testing candidate {index + 1}/{len(candidates)}: {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result = _read_summary(tournament_dir / "summary.txt")
        rows.append(
            {
                "candidate": name,
                "checkpoint": str(checkpoint),
                "epochs": ",".join(str(epoch) for epoch in epochs) or "published-best",
                **result,
            }
        )

    rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate"])))
    with (output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Weight-average screening results",
        f"Games per candidate: {args.battles}",
        f"Common seed: {args.seed}",
        "Action selection: argmax",
        "Candidate and match score:",
    ]
    lines.extend(f"  {row['candidate']}: {float(row['score']):.6f}" for row in rows)
    (output_dir / "scores.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
