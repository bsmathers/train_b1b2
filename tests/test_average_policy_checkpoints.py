from pathlib import Path

import pytest
import torch

from tools.average_policy_checkpoints import average_checkpoints
from tools.test_weight_averages import _remove_restored_best_alias


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


def test_average_policy_checkpoints_recalibrates_sigma_buffers(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "average.pt"
    common = {
        "block.weight": torch.tensor([[3.0, 0.0], [0.0, 1.0]]),
        "block.gamma": torch.ones(1),
    }
    torch.save(
        {**common, "block.u": torch.tensor([1.0, 0.0]), "block.v": torch.tensor([1.0, 0.0])},
        first,
    )
    torch.save(
        {**common, "block.u": torch.tensor([0.0, 1.0]), "block.v": torch.tensor([0.0, 1.0])},
        second,
    )

    average_checkpoints([first, second], [1.0, 1.0], output)
    state = torch.load(output, map_location="cpu", weights_only=True)

    assert torch.linalg.vector_norm(state["block.u"]).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(state["block.v"]).item() == pytest.approx(1.0)
    assert state["block.u"].abs().argmax().item() == 0
    assert state["block.v"].abs().argmax().item() == 0
    manifest = output.with_suffix(".pt.manifest.txt").read_text()
    assert "SigmaReparam layers recalibrated: 1" in manifest


def test_weight_average_screen_removes_only_terminal_best_alias(tmp_path):
    epoch_29 = tmp_path / "policy_epoch_29.pt"
    epoch_30 = tmp_path / "policy_epoch_30.pt"
    published = tmp_path / "policy_actor_bc_final.pt"
    torch.save({"weight": torch.tensor([29.0])}, epoch_29)
    restored_best = {"weight": torch.tensor([17.0])}
    torch.save(restored_best, epoch_30)
    torch.save(restored_best, published)

    filtered, removed = _remove_restored_best_alias(
        [(29, epoch_29), (30, epoch_30)], published
    )

    assert removed == 30
    assert filtered == [(29, epoch_29)]
