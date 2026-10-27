"""Selection checks for the second strong-model actor refinement."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_selected_strong_replays",
    ROOT / "tools" / "prepare_selected_strong_replays.py",
)
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)


class SelectedStrongReplayTests(unittest.TestCase):
    def test_player_parser_uses_perspective_player_only(self):
        name = "gen1ou-1_2200_SmallG1Online-v2_vs_Tauros-v3_01-02-2026_WIN.json.lz4"
        self.assertEqual(selected.trajectory_player(name), "SmallG1Online-v2")
        self.assertIsNone(selected.trajectory_player("not-a-replay.txt"))

    def test_zero_copy_view_selects_exact_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "all" / "gen1ou"
            source.mkdir(parents=True)
            names = (
                "gen1ou-1_2200_Tauros-v2_vs_X_01-02-2026_WIN.json.lz4",
                "gen1ou-2_2200_Tauros-v3_vs_X_01-02-2026_WIN.json.lz4",
                "gen1ou-3_2200_SmallG1Online-v2_vs_X_01-02-2026_WIN.json.lz4",
                "gen1ou-4_2200_SmallG1Online-v3_vs_X_01-02-2026_WIN.json.lz4",
                "gen1ou-5_2200_Tauros-v1_vs_Tauros-v3_01-02-2026_WIN.json.lz4",
            )
            for name in names:
                (source / name).write_bytes(b"trajectory")
            output = root / "selected"
            self.assertEqual(selected.build_view(source.parent, output), 4)
            links = list((output / "gen1ou").iterdir())
            self.assertEqual(len(links), 4)
            self.assertTrue(all(path.is_symlink() for path in links))
            self.assertNotIn(names[-1], {path.name for path in links})


if __name__ == "__main__":
    unittest.main()
