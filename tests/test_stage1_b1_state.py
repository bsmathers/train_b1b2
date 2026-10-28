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
    _monte_carlo_awr_weights,
    _restore_best_state,
    _save_best_state,
    _save_numbered_policy,
)


class _Accelerator:
    is_main_process = True


class Stage1B1StateTests(unittest.TestCase):
    @staticmethod
    def _minimal_agent():
        agent = object.__new__(Stage1B1Agent)
        nn.Module.__init__(agent)
        agent.tstep_encoder = nn.Linear(2, 2)
        agent.traj_encoder = nn.Linear(2, 2)
        agent.actor = nn.Linear(2, 2)
        agent.wld_value = nn.Linear(2, 3)
        agent._b1_phase = Stage1B1Agent.PHASE_VALUE_HEAD
        agent._anchor_loaded = False
        agent._anchor_teacher = None
        agent._phase2_reference = None
        agent._backbone_anchor = None
        agent.policy_kl_coef = 1.0
        agent.actor_policy_kl_coef = 0.25
        return agent

    def test_value_refit_freezes_everything_except_wld_head(self):
        agent = self._minimal_agent()
        agent.set_b1_phase(Stage1B1Agent.PHASE_VALUE_HEAD)
        self.assertTrue(all(p.requires_grad for p in agent.wld_value.parameters()))
        for module in (agent.tstep_encoder, agent.traj_encoder, agent.actor):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_monte_carlo_awr_weights_use_frozen_value_residual(self):
        outcome = torch.tensor([1.0, 1.0, -1.0, -1.0])
        baseline = torch.tensor([0.9, -0.5, -0.9, 0.5], requires_grad=True)
        weights, advantage = _monte_carlo_awr_weights(
            outcome, baseline, temperature=0.5, minimum=0.25, maximum=4.0
        )
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[2]), float(weights[3]))
        self.assertGreaterEqual(float(weights.min()), 0.25)
        self.assertLessEqual(float(weights.max()), 4.0)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(advantage.requires_grad)

    def test_loaded_anchor_is_an_immutable_snapshot(self):
        agent = self._minimal_agent()
        with torch.no_grad():
            agent.actor.weight.fill_(5.0)
        agent.on_checkpoint_loaded(is_resume=False)
        teacher_actor = agent._anchor_teacher[2]
        with torch.no_grad():
            agent.actor.weight.zero_()
        self.assertTrue(torch.equal(teacher_actor.weight, torch.full_like(teacher_actor.weight, 5.0)))
        self.assertTrue(all(not p.requires_grad for p in teacher_actor.parameters()))

    def test_zero_policy_kl_does_not_retain_actor_teacher(self):
        agent = self._minimal_agent()
        agent.policy_kl_coef = 0.0
        agent.actor_policy_kl_coef = 0.0
        agent.on_checkpoint_loaded(is_resume=False)
        self.assertTrue(agent._anchor_loaded)
        self.assertIsNone(agent._anchor_teacher)

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

    def test_numbered_policy_respects_interval_without_training_state(self):
        policy = nn.Linear(2, 1, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            experiment = SimpleNamespace(
                accelerator=_Accelerator(),
                ckpt_dir=tmp,
            )
            _save_numbered_policy(experiment, policy, epoch=9, interval=10)
            self.assertFalse(
                (Path(tmp) / "policy_weights" / "policy_epoch_9.pt").exists()
            )
            _save_numbered_policy(experiment, policy, epoch=10, interval=10)
            self.assertTrue(
                (Path(tmp) / "policy_weights" / "policy_epoch_10.pt").is_file()
            )
            self.assertFalse((Path(tmp) / "training_states").exists())


if __name__ == "__main__":
    unittest.main()
