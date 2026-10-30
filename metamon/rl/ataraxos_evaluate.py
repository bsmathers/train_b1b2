"""Small native promotion evaluator for sequential Ataraxos updates."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    candidate = Path(os.environ["ATARAXOS_EVALUATION_CANDIDATE"]).resolve()
    parent = Path(os.environ["ATARAXOS_EVALUATION_PARENT"]).resolve()
    output_dir = Path(os.environ["ATARAXOS_EVALUATION_DIR"]).resolve()
    result_file = Path(os.environ["ATARAXOS_EVALUATION_RESULT_FILE"]).resolve()
    games = int(os.environ.get("ATARAXOS_EVALUATION_GAMES", "128"))
    score_floor = float(os.environ.get("ATARAXOS_PROMOTION_SCORE_FLOOR", "0.0"))
    max_regression = float(
        os.environ.get("ATARAXOS_PROMOTION_MAX_REGRESSION", "0.02")
    )
    root = Path(__file__).resolve().parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(checkpoint: Path, name: str) -> float:
        destination = output_dir / name
        command = [
            sys.executable,
            str(root / "tools" / "run_b1_vs_taurosv0_500.py"),
            "--checkpoint-path",
            str(checkpoint),
            "--battles",
            str(games),
            "--candidate-team",
            "b1_good_807",
            "--opponent-team",
            "b1_good_807",
            "--save-dir",
            os.environ["ATARAXOS_ARCHITECTURE_SAVE_ROOT"],
            "--run-name",
            os.environ["ATARAXOS_ARCHITECTURE_RUN_NAME"],
            "--cache-dir",
            os.environ.get("METAMON_CACHE_DIR", str(root / "cache")),
            "--output-dir",
            str(destination),
            "--lanes",
            os.environ.get("ATARAXOS_EVALUATION_LANES", "32"),
            "--workers",
            os.environ.get("ATARAXOS_EVALUATION_WORKERS", "4"),
            "--temperature",
            "0",
        ]
        subprocess.run(command, cwd=root, check=True)
        summary = (destination / "summary.txt").read_text(encoding="utf-8")
        match = re.search(
            r"B1 match score with half points for draws/timeouts: ([0-9.]+)",
            summary,
        )
        if match is None:
            raise RuntimeError("native evaluator did not report a match score")
        return float(match.group(1))

    score = evaluate(candidate, "candidate")
    parent_score = evaluate(parent, "parent")
    delta = score - parent_score
    promote = score >= score_floor and delta >= -max_regression
    result_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_file.with_suffix(".tmp")
    temporary.write_text(
        f"promote: {'yes' if promote else 'no'}\n"
        f"score: {score:.6f}\n"
        f"parent_score: {parent_score:.6f}\n"
        f"score_delta: {delta:.6f}\n"
        f"score_floor: {score_floor:.6f}\n"
        f"maximum_allowed_regression: {max_regression:.6f}\n"
        f"games: {games}\n",
        encoding="utf-8",
    )
    os.replace(temporary, result_file)


if __name__ == "__main__":
    main()
