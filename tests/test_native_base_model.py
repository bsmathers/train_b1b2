"""Check the native TaurosV1 starting-weight baseline without downloading weights."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_matchup_baseline", ROOT / "metamon/rl/evaluate/native_matchup.py"
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class FakeTaurosV1:
    pass


class FakeLocalFinetunedModel:
    def __init__(self, **kwargs):
        self.options = kwargs


class NativeBaseModelTests(unittest.TestCase):
    def test_parent_checkpoint_uses_same_cuda_architecture_as_b1(self):
        fake = types.ModuleType("metamon.rl.pretrained")
        fake.LocalFinetunedModel = FakeLocalFinetunedModel
        fake.TaurosV1 = FakeTaurosV1
        fake.get_pretrained_model = lambda model: ("pretrained", model)
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(sys.modules, {"metamon.rl.pretrained": fake}):
                with patch.dict(os.environ, {"TAUROSV1_SAVE_DIR": d}):
                    parent = native.get_candidate_model("TaurosV1", 475)
                    assert isinstance(parent, FakeLocalFinetunedModel)
                    self.assertEqual(parent.options["base_model"], FakeTaurosV1)
                    self.assertEqual(parent.options["amago_ckpt_dir"], d)
                    self.assertEqual(parent.options["model_name"], "mediumg1online")
                    self.assertEqual(parent.options["default_checkpoint"], 475)
                    self.assertEqual(
                        parent.options["model_gin_config"], "grouped_v2_medium_cuda.gin"
                    )
                    self.assertEqual(
                        parent.options["train_gin_config"], "stage1_b1_imitation.gin"
                    )

    def test_baseline_and_finetuning_use_independent_run_directories(self):
        fake = types.ModuleType("metamon.rl.pretrained")
        fake.LocalFinetunedModel = FakeLocalFinetunedModel
        fake.TaurosV1 = FakeTaurosV1
        fake.get_pretrained_model = lambda model: ("pretrained", model)
        with patch.dict(sys.modules, {"metamon.rl.pretrained": fake}):
            with patch.dict(os.environ, {
                "TAUROSV1_SAVE_DIR": "/tmp/original",
                "B1_SAVE_DIR": "/tmp/finetuned",
                "B1_RUN_NAME": "stage1_b1",
            }):
                base = native.get_candidate_model("TaurosV1", 475)
                b1 = native.get_candidate_model("Stage1B1", 1)
                self.assertEqual(base.options["model_name"], "mediumg1online")
                self.assertEqual(base.options["amago_ckpt_dir"], "/tmp/original")
                self.assertEqual(b1.options["model_name"], "stage1_b1")
                self.assertEqual(b1.options["amago_ckpt_dir"], "/tmp/finetuned")
                self.assertEqual(
                    base.options["model_gin_config"],
                    b1.options["model_gin_config"]
                )


if __name__ == "__main__":
    unittest.main()
