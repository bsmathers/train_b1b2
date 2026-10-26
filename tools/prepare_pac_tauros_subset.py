#!/usr/bin/env python3
"""Stream a bounded uniform pac-tauros sample without materializing its full tar."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import tarfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import lz4.frame
import requests
from huggingface_hub import hf_hub_url


REPO_ID = "jakegrigsby/metamon-parsed-pile"
SUBSET = "pac-tauros"
FORMAT = "gen1ou"
COMPLETE_MARKER = ".streamed_subset_complete.txt"
PROGRESS_EVERY = 25_000


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_member_name(member: tarfile.TarInfo) -> PurePosixPath | None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe tar member: {member.name!r}")
    if not member.isfile() or not path.name.endswith((".json", ".json.lz4")):
        return None
    return path


def _destination(root: Path, member_path: PurePosixPath) -> Path:
    # Keep the original basename because Metamon parses outcome metadata from it.
    # A short parent hash prevents collisions without changing that basename.
    bucket = hashlib.sha1(str(member_path).encode("utf-8")).hexdigest()[:2]
    return root / bucket / member_path.name


def _stream_reservoir(
    decompressed_tar,
    output_dir: Path,
    max_trajectories: int,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    selected: list[Path] = []
    selected_paths: set[Path] = set()
    seen = 0

    with tarfile.open(fileobj=decompressed_tar, mode="r|") as archive:
        for member in archive:
            member_path = _safe_member_name(member)
            if member_path is None:
                continue
            seen += 1
            if len(selected) < max_trajectories:
                slot = len(selected)
            else:
                slot = rng.randrange(seen)
                if slot >= max_trajectories:
                    if seen % PROGRESS_EVERY == 0:
                        _log(
                            f"pac-tauros stream: scanned {seen:,}; "
                            f"retained {len(selected):,}"
                        )
                    continue

            destination = _destination(output_dir, member_path)
            if destination in selected_paths:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read tar member: {member.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            try:
                with source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

            if slot < len(selected):
                old = selected[slot]
                selected_paths.remove(old)
                if old != destination:
                    old.unlink(missing_ok=True)
                selected[slot] = destination
            else:
                selected.append(destination)
            selected_paths.add(destination)

            if seen % PROGRESS_EVERY == 0:
                _log(
                    f"pac-tauros stream: scanned {seen:,}; "
                    f"retained {len(selected):,}"
                )

    return seen, len(selected)


def _remove_obsolete_full_archive_files(dataset_root: Path) -> None:
    for name in (
        f"{FORMAT}.tar",
        f"{FORMAT}.tar.lz4",
        f"{FORMAT}.tar.index.sqlite",
        f"{FORMAT}.tar.index.txt",
    ):
        (dataset_root / name).unlink(missing_ok=True)
    # Older hf_hub_download calls may leave a second cached copy of the archive.
    shutil.rmtree(dataset_root / ".cache", ignore_errors=True)
    for path in dataset_root.glob("datasets--jakegrigsby--metamon-parsed-pile*"):
        if path.is_dir():
            shutil.rmtree(path)


def prepare_subset(
    cache_dir: Path,
    max_trajectories: int,
    seed: int,
    revision: str,
    force: bool = False,
) -> Path:
    if max_trajectories < 1:
        raise ValueError("max_trajectories must be positive")
    dataset_root = cache_dir / "self-play" / SUBSET
    output_dir = dataset_root / FORMAT
    marker = output_dir / COMPLETE_MARKER
    if marker.is_file() and not force:
        marker_text = marker.read_text(encoding="utf-8")
        expected_lines = (
            f"Revision: {revision}\n",
            f"Random seed: {seed}\n",
            f"Requested maximum: {max_trajectories}\n",
        )
        if all(line in marker_text for line in expected_lines):
            _log(f"Reusing bounded pac-tauros subset in {output_dir}")
            _remove_obsolete_full_archive_files(dataset_root)
            return output_dir
        raise RuntimeError(
            f"{output_dir} was prepared with different subset settings; "
            "remove it or use --force"
        )
    if output_dir.exists() and not force:
        raise RuntimeError(
            f"{output_dir} exists without a completion marker; remove it or use --force"
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    extracting = dataset_root / f"{FORMAT}.extracting"
    shutil.rmtree(extracting, ignore_errors=True)
    if force:
        shutil.rmtree(output_dir, ignore_errors=True)
    extracting.mkdir(parents=True)

    local_tar = dataset_root / f"{FORMAT}.tar"
    local_lz4 = dataset_root / f"{FORMAT}.tar.lz4"
    try:
        with ExitStack() as stack:
            if local_tar.is_file():
                _log(f"Sampling existing uncompressed archive {local_tar}")
                decompressed = stack.enter_context(local_tar.open("rb"))
            elif local_lz4.is_file():
                _log(f"Streaming existing compressed archive {local_lz4}")
                decompressed = stack.enter_context(lz4.frame.open(local_lz4, "rb"))
            else:
                url = hf_hub_url(
                    repo_id=REPO_ID,
                    filename=f"{SUBSET}/{FORMAT}.tar.lz4",
                    repo_type="dataset",
                    revision=revision,
                )
                headers = {}
                if token := os.environ.get("HF_TOKEN"):
                    headers["Authorization"] = f"Bearer {token}"
                _log(
                    "Streaming pac-tauros directly from Hugging Face; "
                    "the compressed and full tar archives will not be stored"
                )
                response = stack.enter_context(
                    requests.get(url, headers=headers, stream=True, timeout=(30, 300))
                )
                response.raise_for_status()
                content_length = response.headers.get("content-length", "unknown")
                _log(
                    "Connected to pac-tauros archive "
                    f"(compressed bytes reported: {content_length})"
                )
                response.raw.decode_content = False
                decompressed = stack.enter_context(lz4.frame.open(response.raw, "rb"))

            seen, retained = _stream_reservoir(
                decompressed,
                extracting,
                max_trajectories=max_trajectories,
                seed=seed,
            )
        if retained == 0:
            raise RuntimeError("pac-tauros archive contained no replay trajectories")
        (extracting / COMPLETE_MARKER).write_text(
            "Bounded pac-tauros streaming subset\n"
            f"Source repository: {REPO_ID}\n"
            f"Revision: {revision}\n"
            f"Random seed: {seed}\n"
            f"Requested maximum: {max_trajectories}\n"
            f"Source trajectories scanned: {seen}\n"
            f"Trajectories retained: {retained}\n",
            encoding="utf-8",
        )
        extracting.replace(output_dir)
        _remove_obsolete_full_archive_files(dataset_root)
        _log(
            f"Prepared {retained:,} uniformly sampled pac-tauros trajectories "
            f"from {seen:,} source trajectories in {output_dir}"
        )
        return output_dir
    except BaseException:
        shutil.rmtree(extracting, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=475)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_subset(
        cache_dir=args.cache_dir.resolve(),
        max_trajectories=args.max_trajectories,
        seed=args.seed,
        revision=args.revision,
        force=args.force,
    )


if __name__ == "__main__":
    main()
