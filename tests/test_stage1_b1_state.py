"""Focused regression tests for staged B1 checkpoint and L2-SP behavior."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from metamon.rl.stage1_b1 import (
    Stage1B1Agent,
    _restore_best_state,
    _save_best_state,
)


class _Accelerator:
    is_main_process = True


class Stage1B1StateTests(unittest.TestCase):
    def test_best_state_restores_policy_optimizer_and_scheduler(self):
        policy = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / 4, 1.0)
        )
        experiment = SimpleNamespace(
            accelerator=_Accelerator(),
            optimizer=optimizer,
            lr_schedule=scheduler,
            DEVICE=torch.device("cpu"),
        )

        optimizer.zero_grad()
        policy(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        expected_policy = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.pt"
            training_path = Path(tmp) / "training.pt"
            _save_best_state(experiment, policy, policy_path, training_path)

            optimizer.zero_grad()
            policy(torch.full((1, 2), 3.0)).sum().backward()
            optimizer.step()
            scheduler.step()
            # The standalone policy file is an export; restoration must use the
            # single authoritative bundle so an interrupted second write cannot
            # pair one policy candidate with another optimizer state.
            torch.save({"weight": torch.full_like(policy.weight, 99.0)}, policy_path)
            _restore_best_state(experiment, policy, training_path)

        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, expected_policy[name]))
        restored_optimizer = optimizer.state_dict()
        self.assertEqual(restored_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter_id, state in expected_optimizer["state"].items():
            for key, expected in state.items():
                restored = restored_optimizer["state"][parameter_id][key]
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(restored, expected))
                else:
                    self.assertEqual(restored, expected)
        self.assertEqual(scheduler.state_dict(), expected_scheduler)

    def test_l2sp_is_parameter_normalized_and_prefix_weighted(self):
        tstep = nn.Linear(2, 2, bias=False)
        traj = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            tstep.weight.fill_(1.0)
            traj.weight.fill_(2.0)
        anchor = {
            "tstep_encoder.weight": torch.zeros_like(tstep.weight),
            "traj_encoder.weight": torch.zeros_like(traj.weight),
        }
        fake = SimpleNamespace(
            tstep_encoder=tstep,
            traj_encoder=traj,
            _backbone_anchor=anchor,
            l2sp_layer_weights={},
        )
        self.assertAlmostEqual(float(Stage1B1Agent._l2sp(fake).detach()), 2.0)

        fake.l2sp_layer_weights = {
            "tstep_encoder": 2.0,
            "traj_encoder": 0.5,
        }
        self.assertAlmostEqual(
            float(Stage1B1Agent._l2sp(fake).detach()), 12.0 / 9.0
        )

        # The longest component-aligned prefix wins. This lets a single layer
        # override its encoder default without accidentally matching layer 10.
        fake.l2sp_layer_weights["tstep_encoder.weight"] = 4.0
        self.assertAlmostEqual(
            float(Stage1B1Agent._l2sp(fake).detach()), 20.0 / 17.0
        )


if __name__ == "__main__":
    unittest.main()
