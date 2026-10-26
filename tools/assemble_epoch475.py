"""Reassemble and verify the repository-sharded TaurosV1 Epoch475 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


EXPECTED_BYTES = 139_252_367
EXPECTED_SHA256 = "f7a8f40575127b36350b92f78f01f7a1c517ab4b9878a505fd4269baef844e54"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    shard_dir = root / "artifacts" / "epoch475"
    shards = sorted(shard_dir.glob("policy_epoch_475.pt.part-*"))
    if not shards:
        raise FileNotFoundError(f"no Epoch475 shards found in {shard_dir}")

    target = (
        root
        / "checkpoints"
        / "mediumg1online"
        / "ckpts"
        / "policy_weights"
        / "policy_epoch_475.pt"
    )
    if target.is_file() and target.stat().st_size == EXPECTED_BYTES:
        if sha256(target) == EXPECTED_SHA256:
            print(f"Epoch475 checkpoint already verified: {target}")
            return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as output:
        for shard in shards:
            with shard.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)

    actual_size = temporary.stat().st_size
    actual_sha256 = sha256(temporary)
    if actual_size != EXPECTED_BYTES or actual_sha256 != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Epoch475 shard verification failed: "
            f"bytes={actual_size}, sha256={actual_sha256}"
        )
    os.replace(temporary, target)
    print(f"Reassembled and verified Epoch475 checkpoint: {target}")


if __name__ == "__main__":
    main()
