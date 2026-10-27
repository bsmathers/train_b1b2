#!/usr/bin/env python3
"""Create a zero-copy replay view for the selected strong-policy actors."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_PLAYERS = (
    "Tauros-v1",
    "Tauros-v2",
    "SmallG1Online-v2",
    "SmallG1Online-v3",
)


def trajectory_player(filename: str) -> str | None:
    """Return the perspective player encoded by a parsed-replay filename."""
    if filename.endswith(".json.lz4"):
        stem = filename[: -len(".json.lz4")]
    elif filename.endswith(".json"):
        stem = filename[: -len(".json")]
    else:
        return None
    parts = stem.split("_")
    if len(parts) < 7:
        return None
    vs_positions = [i for i in range(2, len(parts) - 2) if parts[i] == "vs"]
    if len(vs_positions) != 1:
        return None
    return "_".join(parts[2 : vs_positions[0]])


def select_trajectories(source_dir: Path, players: tuple[str, ...]):
    allowed = set(players)
    counts = Counter({player: 0 for player in players})
    selected = []
    malformed = 0
    with os.scandir(source_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            player = trajectory_player(entry.name)
            if player is None:
                if entry.name.endswith((".json", ".json.lz4")):
                    malformed += 1
                continue
            if player in allowed:
                counts[player] += 1
                selected.append(Path(entry.path))
    return selected, counts, malformed


def print_counts(counts: Counter, malformed: int) -> None:
    print("Selected strong-model trajectory counts")
    for player in DEFAULT_PLAYERS:
        print(f"  {player}: {counts[player]:,}")
    print(f"  Total: {sum(counts.values()):,}")
    print(f"  Malformed replay filenames ignored: {malformed:,}")


def build_view(source_root: Path, output_root: Path, count_only: bool = False) -> int:
    source_dir = source_root / "gen1ou"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"strong replay directory was not found: {source_dir}")
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source and output replay roots must differ")

    selected, counts, malformed = select_trajectories(source_dir, DEFAULT_PLAYERS)
    print_counts(counts, malformed)
    if malformed:
        raise RuntimeError("refusing to build a subset from malformed replay filenames")
    if not selected:
        raise RuntimeError("no selected strong-model trajectories were found")
    if count_only:
        return len(selected)

    ready = output_root / "SOURCE.txt"
    if ready.is_file() and (output_root / "gen1ou").is_dir():
        expected = f"Trajectory files: {len(selected)}\n"
        if expected in ready.read_text(encoding="utf-8"):
            print(f"Reusing selected replay view in {output_root / 'gen1ou'}")
            return len(selected)
        raise RuntimeError(
            f"existing selected replay view is stale: {output_root}; remove that exact "
            "generated directory and retry"
        )
    if output_root.exists():
        raise RuntimeError(f"refusing to replace existing path: {output_root}")

    staging = output_root.with_name(output_root.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    destination_dir = staging / "gen1ou"
    destination_dir.mkdir(parents=True)
    try:
        for source in selected:
            destination = destination_dir / source.name
            destination.symlink_to(os.path.relpath(source, destination_dir))
        lines = [
            "Selected strong-model replay view",
            f"Source: {source_root.resolve()}",
            "Perspective players: " + ", ".join(DEFAULT_PLAYERS),
        ]
        lines.extend(f"{player}: {counts[player]}" for player in DEFAULT_PLAYERS)
        lines.append(f"Trajectory files: {len(selected)}")
        (staging / "SOURCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared {len(selected):,} selected trajectories in {output_root / 'gen1ou'}")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    if not args.count_only and args.output is None:
        parser.error("--output is required unless --count-only is used")
    output = args.output or args.source.with_name(args.source.name + ".unused")
    build_view(args.source.expanduser(), output.expanduser(), args.count_only)


if __name__ == "__main__":
    main()
