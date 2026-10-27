from pathlib import Path

import pytest
import torch

from tools.average_policy_checkpoints import average_checkpoints


def _save(path: Path, weight: float, count: int = 3) -> None:
    torch.save(
        {
            "layer.weight": torch.tensor([weight, weight + 2], dtype=torch.bfloat16),
            "position_ids": torch.arange(count),
        },
        path,
    )


def test_average_policy_checkpoints_preserves_dtype_and_integer_state(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "average.pt"
    _save(first, 1.0)
    _save(second, 3.0)

    average_checkpoints([first, second], [1.0, 3.0], output)
    state = torch.load(output, map_location="cpu", weights_only=True)

    assert state["layer.weight"].dtype == torch.bfloat16
    assert torch.equal(state["layer.weight"].float(), torch.tensor([2.5, 4.5]))
    assert torch.equal(state["position_ids"], torch.arange(3))
    assert output.with_suffix(".pt.manifest.txt").is_file()


def test_average_policy_checkpoints_rejects_different_integer_state(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _save(first, 1.0, count=3)
    _save(second, 3.0, count=4)

    with pytest.raises(ValueError, match="incompatible tensor"):
        average_checkpoints([first, second], [1.0, 1.0], tmp_path / "average.pt")
