import unittest

from metamon.env.vectorized.native_binary import (
    GEN1_BATTLE_BYTES,
    RESULT_WIN,
    NativeBinaryLane,
    result_type,
)


class NativeBinaryLaneTest(unittest.TestCase):
    def test_alignment_and_terminal_result(self):
        lane = NativeBinaryLane(3)
        lane.begin({"name": "p1"}, {"name": "p2"})
        state = bytes(GEN1_BATTLE_BYTES)
        lane.handle_native_state(state, 0x50, b"\x05\x09", b"\x06")
        lane.commit(0x05, 0x06)
        lane.handle_native_state(state, RESULT_WIN, b"", b"")
        self.assertTrue(lane.terminal)
        self.assertEqual(lane.trajectory.result, RESULT_WIN)
        self.assertEqual(len(lane.trajectory.states), 2)
        self.assertEqual(len(lane.trajectory.actions), 1)

    def test_result_type_ignores_request_bits(self):
        self.assertEqual(result_type(0xA1), RESULT_WIN)

    def test_rejects_wrong_battle_size(self):
        lane = NativeBinaryLane(0)
        lane.begin({}, {})
        with self.assertRaisesRegex(RuntimeError, "expected 384"):
            lane.handle_native_state(b"short", 0, b"", b"")


if __name__ == "__main__":
    unittest.main()
