#!/usr/bin/env python3
"""Run one exact, side-balanced B1-vs-TaurosV0 native match.

The requested total is split between B1 as physical p1 and physical p2.  Raw
wins, decisive win rate, natural draws, timeouts, and half-point match score
are reported separately so "win rate" cannot silently mean several things.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_result(path: Path, battles: int, side: int) -> dict:
    data = json.loads(path.read_text())
    required = {
        "battles",
        "candidate_wins",
        "opponent_wins",
        "draws",
        "natural_draws",
        "timeouts",
        "candidate_score",
        "engine",
        "side",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"result is missing fields {sorted(missing)}: {path}")
    if data["engine"] != "zig-pkmn" or data["side"] != side:
        raise ValueError(f"wrong engine or physical side in {path}")
    counts = [
        int(data["candidate_wins"]),
        int(data["opponent_wins"]),
        int(data["natural_draws"]),
        int(data["timeouts"]),
    ]
    if int(data["battles"]) != battles or min(counts) < 0 or sum(counts) != battles:
        raise ValueError(f"invalid or incomplete outcome counts in {path}")
    if int(data["draws"]) != counts[2] + counts[3]:
        raise ValueError(f"draw count does not equal natural draws plus timeouts: {path}")
    expected_score = (counts[0] + 0.5 * (counts[2] + counts[3])) / battles
    if abs(float(data["candidate_score"]) - expected_score) > 1e-12:
        raise ValueError(f"candidate score disagrees with outcome counts: {path}")
    return data


def _configuration(
    args, per_side: list[int], checkpoint_source: Path, checkpoint_sha256: str
) -> str:
    return "\n".join(
        (
            "B1 versus TaurosV0 native match",
            f"Candidate checkpoint source: {checkpoint_source}",
            f"Candidate checkpoint SHA-256: {checkpoint_sha256}",
            f"Candidate team set: {args.candidate_team}",
            f"TaurosV0 team set: {args.opponent_team}",
            f"Total games: {args.battles}",
            f"B1 as p1: {per_side[0]}",
            f"B1 as p2: {per_side[1]}",
            f"Seed: {args.seed}",
            f"Action temperature: {args.temperature:g}",
            f"Action selection: {'argmax' if args.temperature == 0 else 'sample'}",
            f"Decision limit per game: {args.turn_limit}",
            f"Vector lanes: {args.lanes}",
            f"Simulator workers: {args.workers}",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=int)
    checkpoint.add_argument("--checkpoint-path")
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--candidate-team", default="b1_good_807")
    parser.add_argument("--opponent-team", default="b1_good_807")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--run-name", default="stage1_b1_actor_bc")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Action temperature for both policies; zero means deterministic argmax.",
    )
    parser.add_argument("--turn-limit", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if min(args.battles, args.lanes, args.workers, args.turn_limit, args.timeout) < 1:
        parser.error("battles, lanes, workers, turn-limit, and timeout must be positive")
    if args.temperature < 0:
        parser.error("temperature must be non-negative")
    if args.checkpoint_path and not Path(args.checkpoint_path).expanduser().is_file():
        parser.error(f"checkpoint path does not exist: {args.checkpoint_path}")

    args.save_dir = args.save_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_side = [(args.battles + 1) // 2, args.battles // 2]

    checkpoint_source = (
        Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else args.save_dir
        / args.run_name
        / "ckpts"
        / "policy_weights"
        / f"policy_epoch_{args.checkpoint}.pt"
    )
    if not checkpoint_source.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_source}")

    os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir)
    os.environ["B1_SAVE_DIR"] = str(args.save_dir)
    os.environ["B1_RUN_NAME"] = args.run_name
    # Stage1B1 derives its observation/action metadata from the local TaurosV1
    # parent, so constructing the evaluator still needs the Epoch475 directory.
    os.environ.setdefault("TAUROSV1_SAVE_DIR", str(args.save_dir))
    if "b1_good_807" in (args.candidate_team, args.opponent_team):
        from tools.select_stage1_b1_epoch import install_good_teams

        install_good_teams(args.cache_dir)

    # The live best checkpoint can be atomically replaced by training between
    # physical-side runs. Snapshot it once so all 500 games use identical bits.
    checkpoint_snapshot = args.output_dir / "candidate_checkpoint.pt"
    if checkpoint_snapshot.is_file():
        checkpoint_sha256 = _sha256(checkpoint_snapshot)
    else:
        temporary = checkpoint_snapshot.with_suffix(".tmp")
        shutil.copy2(checkpoint_source, temporary)
        os.replace(temporary, checkpoint_snapshot)
        checkpoint_sha256 = _sha256(checkpoint_snapshot)

    config_path = args.output_dir / "configuration.txt"
    config = _configuration(args, per_side, checkpoint_source, checkpoint_sha256)
    if config_path.exists() and config_path.read_text() != config:
        raise RuntimeError(
            f"existing output has a different configuration: {config_path}; "
            "choose a new output directory"
        )
    config_path.write_text(config)

    results = []
    game_log_path = args.output_dir / "games.log"
    for side, battles in enumerate(per_side):
        if battles == 0:
            continue
        result_path = args.output_dir / f"side{side}_result.json"
        log_path = args.output_dir / f"side{side}.log"
        if result_path.is_file():
            results.append(_read_result(result_path, battles, side))
            print(f"Reusing validated side {side} result: {result_path}", flush=True)
            continue
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "metamon.rl.evaluate.native_matchup",
            "--model",
            "Stage1B1",
            "--candidate-team",
            args.candidate_team,
            "--opponent-team",
            args.opponent_team,
            "--side",
            str(side),
            "--battles",
            str(battles),
            "--lanes",
            str(min(args.lanes, battles)),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--temperature",
            str(args.temperature),
            "--turn-limit",
            str(args.turn_limit),
            "--output",
            str(result_path),
            "--game-log",
            str(game_log_path),
        ]
        cmd.extend(["--checkpoint-path", str(checkpoint_snapshot)])
        print(
            f"Running {battles} games with B1 as p{side + 1}; log: {log_path}",
            flush=True,
        )
        env = os.environ.copy()
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        with log_path.open("w") as log:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"side {side} failed; inspect {log_path}")
        results.append(_read_result(result_path, battles, side))

    wins = sum(int(r["candidate_wins"]) for r in results)
    losses = sum(int(r["opponent_wins"]) for r in results)
    natural_draws = sum(int(r["natural_draws"]) for r in results)
    timeouts = sum(int(r["timeouts"]) for r in results)
    decisive = wins + losses
    raw_win_rate = wins / args.battles
    decisive_win_rate = wins / decisive if decisive else float("nan")
    match_score = (wins + 0.5 * (natural_draws + timeouts)) / args.battles
    lines = [
        "B1 versus TaurosV0 result",
        f"Total games: {args.battles}",
        f"B1 wins: {wins}",
        f"TaurosV0 wins: {losses}",
        f"Natural draws: {natural_draws}",
        f"Decision-limit timeouts: {timeouts}",
        f"Raw B1 win rate: {raw_win_rate:.6f}",
        f"B1 win rate among decisive games: {decisive_win_rate:.6f}",
        f"B1 match score with half points for draws/timeouts: {match_score:.6f}",
    ]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
