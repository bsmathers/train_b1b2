"""Verify automatic Zig turns have a single canonical policy action."""
import importlib.util
import unittest
from pathlib import Path

source = (Path(__file__).resolve().parents[1] /
          "metamon/env/vectorized/native_legal.py")
spec = importlib.util.spec_from_file_location("native_legal", source)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

class NativeLegalTests(unittest.TestCase):
    def test_forced_recharge_and_rage(self):
        for pseudo in ("recharge", "fight"):
            req = {"active": [{"moves": [{"id": pseudo}]}]}
            self.assertEqual(native.canonical_legal_indices(list(range(9)), req), [0])

    def test_struggle_preserves_switches_but_not_extra_move_logits(self):
        req = {"active": [{"moves": [{"id": "struggle"}]}]}
        self.assertEqual(
            native.canonical_legal_indices([0, 1, 2, 3, 4, 7], req),
            [0, 4, 7],
        )

    def test_normal_moves_unchanged(self):
        req = {"active": [{"moves": [{"id": "blizzard"}]}]}
        self.assertEqual(native.canonical_legal_indices([3, 5], req), [3, 5])
        self.assertEqual(native.canonical_legal_indices([0, 2, 4], None), [0, 2, 4])

if __name__ == "__main__":
    unittest.main()
