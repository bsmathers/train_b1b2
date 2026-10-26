#!/usr/bin/env python3
"""Choose a Stage-1 B1 epoch by balanced H2H vs TaurosV0.

The four B1/TaurosV0 team assignments are good/good, good/bad,
bad/good and bad/bad. Each is run in both challenger/acceptor orders.
The reported macro win rate gives every assignment equal weight.
"""
import argparse
import csv
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from metamon.rl.evaluate.common import PolicySpec, MatchupSpec
from metamon.rl.evaluate.launch import run_all_matchups
from metamon.rl.evaluate.results import ResultsTracker

CONDITIONS = (("good", "good"), ("good", "bad"),
              ("bad", "good"), ("bad", "bad"))


def install_good_teams(archive: Path, cache: Path) -> int:
    target = cache / "teams" / "b1_good_807" / "gen1ou"
    target.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(f"Good-team ZIP not found: {archive}")
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist()
                         if n.startswith("good_teams/")
                         and n.endswith((".txt", ".gen1ou_team"))
                         and not n.startswith("__MACOSX/"))
        if len(members) != 807:
            raise ValueError(f"Expected 807 Showdown teams, found {len(members)}")
        for i, member in enumerate(members):
            data = z.read(member)
            if not data.strip():
                raise ValueError(f"Empty team: {member}")
            # TeamSet discovers files by format-specific extension.
            (target / f"team_{i:04d}.gen1ou_team").write_bytes(data)
    return len(members)


def collect_epochs(save_dir: Path, run_name: str):
    base = save_dir / run_name / "ckpts" / "policy_weights"
    epochs = []
    for path in base.glob("policy_epoch_*.pt"):
        match = re.fullmatch(r"policy_epoch_(\d+)\.pt", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def specs(epochs, battles, good_set, bad_set):
    pools = {"good": good_set, "bad": bad_set}
    for epoch in epochs:
        for b1_pool, v0_pool in CONDITIONS:
            for reverse in (False, True):
                b1 = PolicySpec(
                    name=f"B1_e{epoch}_{b1_pool}",
                    model_name="Stage1B1", checkpoint=epoch,
                    temperature=1.0, team_set=pools[b1_pool],
                    battle_backend="metamon",
                )
                v0 = PolicySpec(
                    name=f"V0_{v0_pool}",
                    model_name="TaurosV0", checkpoint=62,
                    temperature=1.0, team_set=pools[v0_pool],
                    battle_backend="metamon",
                )
                first, second = (v0, b1) if reverse else (b1, v0)
                yield (epoch, b1_pool, v0_pool, reverse,
                       MatchupSpec(first, second, battles, "gen1ou"))


def summarize(entries, results, out):
    completed = {r.matchup_id: r for r in results}
    rows = []
    for epoch in sorted({e[0] for e in entries}):
        conditions = []
        valid = True
        for bp, vp in CONDITIONS:
            side_rates = []
            for entry in entries:
                e, ebp, evp, rev, match = entry
                if (e, ebp, evp) != (epoch, bp, vp):
                    continue
                r = completed.get(match.matchup_id)
                if r is None or r.total_battles != match.n_battles:
                    valid = False
                    break
                wins = r.policy_b_wins if rev else r.policy_a_wins
                # Count ties as 1/2 (the result CSV counts draws in total).
                draws = r.total_battles - r.policy_a_wins - r.policy_b_wins
                side_rates.append((wins + 0.5 * draws) / r.total_battles)
            if len(side_rates) != 2:
                valid = False
                conditions.append(None)
            else:
                conditions.append(sum(side_rates) / 2)
        rows.append({"epoch": epoch, "complete": valid,
                     **{f"{a}_{b}": r for (a, b), r in zip(CONDITIONS, conditions)},
                     "macro_win_rate": (sum(conditions) / 4 if valid else None)})
    with open(out / "epoch_selection.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    complete = [r for r in rows if r["complete"]]
    if not complete:
        print("No fully completed epochs; see missing/partial matchup results.")
        return
    best = max(complete, key=lambda r: (r["macro_win_rate"], -r["epoch"]))
    (out / "selected_epoch.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"Selected epoch {best['epoch']}: macro win rate {best['macro_win_rate']:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--good-teams-zip", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", default="stage1_b1")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/b1_selection"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--gpus", nargs="+", type=int, default=[0])
    p.add_argument("--battles-per-side", type=int, default=50)
    p.add_argument("--epochs", nargs="*", type=int)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.battles_per_side < 1 or args.max_concurrent < 1:
        p.error("battle count and concurrency must be positive")
    count = install_good_teams(args.good_teams_zip, args.cache_dir.resolve())
    os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir.resolve())
    os.environ["B1_SAVE_DIR"] = str(args.save_dir.resolve())
    os.environ["B1_RUN_NAME"] = args.run_name
    epochs = args.epochs if args.epochs else collect_epochs(args.save_dir, args.run_name)
    missing = [e for e in epochs if not (args.save_dir / args.run_name /
               "ckpts" / "policy_weights" / f"policy_epoch_{e}.pt").is_file()]
    if missing or not epochs:
        p.error(f"No usable B1 epochs; missing: {missing}")
    rows = list(specs(epochs, args.battles_per_side, "b1_good_807", "hl_05_26"))
    print(f"Using {count} good teams, hl_05_26 bad teams, epochs={epochs}; {len(rows)} matchups")
    if args.dry_run:
        for e, a, b, rev, m in rows:
            print(f"epoch={e} {a}/{b} {'reversed' if rev else 'normal'}: {m.matchup_id}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all_matchups([r[-1] for r in rows], gpus=args.gpus,
                     output_dir=str(args.output_dir),
                     max_concurrent=args.max_concurrent)
    tracker = ResultsTracker(str(args.output_dir))
    summarize(rows, tracker.get_all_results(), args.output_dir)


if __name__ == "__main__":
    main()
