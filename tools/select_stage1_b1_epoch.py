#!/usr/bin/env python3
"""B1/B2 checkpoint selection using balanced Gen1 OU tournaments vs TaurosV0.

B1: Evaluate every saved checkpoint. B2: interpolate top B1 checkpoints with
selected TaurosV1 base checkpoints at multiple alphas; tournament the resulting
models too. Each candidate plays good/good, good/bad, bad/good and bad/bad,
in both challenger and acceptor roles. Draws count as half a win.
"""

import argparse
import base64
import bz2
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

GOOD_SHA256 = "b26863a9142c8da7a5cfad3989b461d940ad71c7c768536d0f7bdd87c87860cb"
ROOT = Path(__file__).resolve().parents[1]
PACKED_GOOD = ROOT / "artifacts" / "good_teams_807.bz2.base64"
CONDITIONS = (("good", "good"), ("good", "bad"), ("bad", "good"), ("bad", "bad"))


def good_team_bytes(archive=None):
    if archive:
        with zipfile.ZipFile(archive) as src:
            names = sorted(
                n
                for n in src.namelist()
                if n.startswith("good_teams/") and n.endswith((".txt", ".gen1ou_team"))
            )
            teams = [src.read(n) for n in names]
    else:
        packed = base64.b64decode(PACKED_GOOD.read_text().strip(), validate=True)
        raw = bz2.decompress(packed)
        if hashlib.sha256(raw).hexdigest() != GOOD_SHA256:
            raise ValueError("Bundled good-team archive failed SHA-256 verification")
        teams = raw.split(b"\x00")
    if len(teams) != 807 or any(b"- " not in team for team in teams):
        raise ValueError("Expected exactly 807 nonempty Showdown-export teams")
    return teams


def install_good_teams(cache_dir, archive=None):
    teams = good_team_bytes(archive)
    destination = cache_dir / "teams" / "b1_good_807" / "gen1ou"
    if destination.exists():
        found = sorted(destination.glob("*.gen1ou_team"))
        if len(found) == len(teams) and all(
            p.read_bytes() == team for p, team in zip(found, teams)
        ):
            return destination
        raise ValueError(
            f"Existing team directory does not match bundled data: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="good_807_", dir=destination.parent))
    try:
        for i, team in enumerate(teams):
            (temporary / f"team_{i:04d}.gen1ou_team").write_bytes(team)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def saved_epochs(save_dir, run_name):
    directory = save_dir / run_name / "ckpts" / "policy_weights"
    return sorted(
        int(m.group(1))
        for f in directory.glob("policy_epoch_*.pt")
        if (m := re.fullmatch(r"policy_epoch_(\d+)\.pt", f.name))
        and int(m.group(1)) > 0
    )


def policy(name, model, ckpt, pool):
    from metamon.rl.evaluate.common import PolicySpec

    return PolicySpec(
        name=name,
        model_name=model,
        checkpoint=ckpt,
        temperature=1.0,
        team_set=("b1_good_807" if pool == "good" else "hl_05_26"),
        battle_backend="metamon",
    )


def matchups(candidate, model, ckpt, battles):
    from metamon.rl.evaluate.common import MatchupSpec

    for own, other in CONDITIONS:
        for reverse in (False, True):
            trained = policy(f"{candidate}_{own}", model, ckpt, own)
            baseline = policy(f"V0_{other}", "TaurosV0", 62, other)
            first, second = (baseline, trained) if reverse else (trained, baseline)
            spec = MatchupSpec(first, second, battles, "gen1ou")
            yield (candidate, own, other, reverse, spec)


def validate_manifest(outdir, manifest):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "tournament_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(
                f"Configuration changed for existing results: {path}. "
                "Choose a new output directory."
            )
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def score_tournament(entries, results, output):
    completed = {r.matchup_id: r for r in results}
    rows = []
    for candidate in dict.fromkeys(e[0] for e in entries):
        conditions = {}
        for own, other in CONDITIONS:
            rates = []
            for cand, a, b, reverse, spec in entries:
                if (cand, a, b) != (candidate, own, other):
                    continue
                result = completed.get(spec.matchup_id)
                if result is None or result.total_battles != spec.n_battles:
                    continue
                draws = (
                    result.total_battles - result.policy_a_wins - result.policy_b_wins
                )
                if draws < 0:
                    continue
                wins = result.policy_b_wins if reverse else result.policy_a_wins
                rates.append((wins + 0.5 * draws) / result.total_battles)
            conditions[f"{own}_{other}"] = sum(rates) / 2 if len(rates) == 2 else None
        done = all(v is not None for v in conditions.values())
        rows.append(
            {
                "candidate": candidate,
                "complete": done,
                **conditions,
                "macro_win_rate": (sum(conditions.values()) / 4 if done else None),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def ranked_completed(rows):
    return sorted(
        (r for r in rows if r["complete"]),
        key=lambda r: (-r["macro_win_rate"], r["candidate"]),
    )


def run_tournament(entries, output, args):
    from metamon.rl.evaluate.launch import run_all_matchups
    from metamon.rl.evaluate.results import ResultsTracker

    manifest = {
        "battles_per_side": args.battles_per_side,
        "good_sha256": GOOD_SHA256,
        "matchups": [r[-1].matchup_id for r in entries],
    }
    validate_manifest(output, manifest)
    # The generic H2H tracker considers any nonempty battle log complete.
    # For model selection, partially completed bouts must be retried, and
    # prior CSVs must not be appended to on retries.
    results_file = output / "matchup_results.jsonl"
    existing = []
    if results_file.exists():
        existing = [
            json.loads(line)
            for line in results_file.read_text().splitlines()
            if line.strip()
        ]
    required = {e[-1].matchup_id: e[-1].n_battles for e in entries}
    valid = {
        r["matchup_id"]
        for r in existing
        if r.get("total_battles") == required.get(r["matchup_id"])
    }
    if not args.dry_run:
        results_file.write_text(
            "".join(json.dumps(r) + "\n" for r in existing if r["matchup_id"] in valid)
        )
        for matchup_id in required:
            if matchup_id not in valid:
                partial_dir = output / matchup_id
                if partial_dir.is_dir():
                    shutil.rmtree(partial_dir)
    if args.dry_run:
        print(
            f"Would run {len(entries)} matchups, {args.battles_per_side} battles each"
        )
        for _, _, _, _, spec in entries:
            print(spec.matchup_id)
        return []
    run_all_matchups(
        [r[-1] for r in entries],
        gpus=args.gpus,
        output_dir=str(output),
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    tracker = ResultsTracker(str(output))
    return score_tournament(entries, tracker.get_all_results(), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("b1", "b2", "both"), default="both")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--taurosv1-dir", type=Path)
    parser.add_argument("--run-name", default="stage1_b1")
    parser.add_argument("--b2-run-name", default="stage1_b2_selection")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "checkpoints" / "stage1_selection"
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--good-teams-zip", type=Path, help="Optional original ZIP override"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--epochs", type=int, nargs="*", help="B1 epochs (default: all saved)"
    )
    parser.add_argument("--base-epochs", type=int, nargs="+", default=[475])
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--battles-per-side", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.battles_per_side < 1 or args.top_k < 1 or args.max_concurrent < 1:
        parser.error("battle count, top-k and concurrency must be positive")
    if not args.gpus or any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        parser.error("At least one GPU and all alphas in [0,1] are required")
    args.save_dir = args.save_dir.resolve()
    args.taurosv1_dir = (args.taurosv1_dir or args.save_dir).resolve()
    args.cache_dir = args.cache_dir.resolve()
    os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir)
    os.environ["TAUROSV1_SAVE_DIR"] = str(args.taurosv1_dir)
    os.environ["B1_SAVE_DIR"] = str(args.save_dir)
    os.environ["B1_RUN_NAME"] = args.run_name
    os.environ["B2_SAVE_DIR"] = str(args.save_dir)
    os.environ["B2_RUN_NAME"] = args.b2_run_name
    install_good_teams(args.cache_dir, args.good_teams_zip)
    epochs = sorted(
        set(
            args.epochs
            if args.epochs is not None
            else saved_epochs(args.save_dir, args.run_name)
        )
    )
    if not epochs:
        parser.error("No B1 epoch checkpoints found")
    for epoch in epochs:
        path = (
            args.save_dir
            / args.run_name
            / "ckpts"
            / "policy_weights"
            / f"policy_epoch_{epoch}.pt"
        )
        if epoch < 1 or not path.is_file():
            parser.error(f"Missing valid B1 checkpoint: {path}")
    if args.mode in ("b1", "both"):
        b1_entries = [
            entry
            for epoch in epochs
            for entry in matchups(
                f"B1_e{epoch}", "Stage1B1", epoch, args.battles_per_side
            )
        ]
        b1_scores = run_tournament(b1_entries, args.output_dir / "b1", args)
        if args.dry_run:
            return
        ranked = ranked_completed(b1_scores)
        if len(ranked) != len(epochs):
            raise RuntimeError(
                "B1 tournament results incomplete; rerun to finish before B2"
            )
        (args.output_dir / "b1" / "selected_epoch.json").write_text(
            json.dumps(ranked[0], indent=2) + "\n"
        )
        b1_choice = [int(r["candidate"].split("_e")[-1]) for r in ranked[: args.top_k]]
        print(f"Selected B1 epoch {b1_choice[0]}")
        if args.mode == "b1":
            return
    else:
        b1_scores_path = args.output_dir / "b1" / "candidate_scores.csv"
        if not b1_scores_path.exists():
            parser.error("B1 results missing; run --mode b1 first")
        with b1_scores_path.open(newline="") as f:
            ranked = ranked_completed(
                [
                    {
                        **row,
                        "complete": row["complete"] == "True",
                        "macro_win_rate": (
                            float(row["macro_win_rate"])
                            if row["macro_win_rate"]
                            else None
                        ),
                    }
                    for row in csv.DictReader(f)
                ]
            )
        b1_choice = [int(r["candidate"].split("_e")[-1]) for r in ranked[: args.top_k]]
        if not b1_choice:
            parser.error("No completed B1 candidates")
    from metamon.rl.stage1_b2_select import generate_interpolations

    combinations = [
        (base, b1, alpha)
        for base in sorted(set(args.base_epochs))
        for b1 in b1_choice
        for alpha in sorted(set(args.alphas))
        if alpha != 0.0 or b1 == b1_choice[0]
    ]
    b2_manifest = generate_interpolations(
        combinations,
        args.taurosv1_dir,
        args.save_dir,
        args.run_name,
        args.b2_run_name,
        dry_run=args.dry_run,
    )
    b2_entries = [
        item
        for info in b2_manifest
        for item in matchups(
            f"B2_base{info['base_epoch']}_b1{info['b1_epoch']}_a{info['alpha']}",
            "Stage1B2",
            info["id"],
            args.battles_per_side,
        )
    ]
    b2_scores = run_tournament(b2_entries, args.output_dir / "b2", args)
    if not args.dry_run:
        ranked_b2 = ranked_completed(b2_scores)
        if len(ranked_b2) != len(b2_manifest):
            raise RuntimeError("B2 tournament incomplete; rerun to finish")
        selected = ranked_b2[0]
        info = next(
            info
            for info in b2_manifest
            if selected["candidate"]
            == f"B2_base{info['base_epoch']}_b1{info['b1_epoch']}_a{info['alpha']}"
        )
        final = {**selected, **info}
        (args.output_dir / "b2" / "selected_interpolation.json").write_text(
            json.dumps(final, indent=2) + "\n"
        )
        print("Selected B2:", json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
