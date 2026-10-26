"""Hermetic regression test for native Zig setup's build working directory.

This intentionally mocks node/zig/npm: no network access or compiler needed.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeSetupTests(unittest.TestCase):
    def test_install_from_package_root_and_skip_rebuild(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools_dir = root / "tools"
            host_dir = root / "metamon" / "env" / "vectorized"
            engine = host_dir / "node_modules" / "@pkmn" / "engine"
            (engine / "src" / "bin").mkdir(parents=True)
            (engine / "src" / "bin" / "install-pkmn-engine").write_text(
                "# mock upstream driver\n"
            )
            (engine / "build" / "pkg").mkdir(parents=True)
            (host_dir / "pkmn_engine_common.js").write_text("// mock common\n")
            tools_dir.mkdir()
            shutil.copy2(
                repository / "tools" / "setup_pkmn_engine.sh",
                tools_dir / "setup_pkmn_engine.sh",
            )
            binaries = root / "fake_bin"
            binaries.mkdir()
            build_log = root / "builds.log"
            # The fake upstream installer only works from the npm package
            # directory, mirroring the real relative rename destination.
            node = binaries / "node"
            node.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "from pathlib import Path\n"
                "argv = sys.argv[1:]\n"
                "if argv == ['--version']:\n"
                "    print('v20.20.2')\n"
                "elif argv and argv[0] == '-':\n"
                "    sys.exit(0 if Path(argv[1]).is_file() else 1)\n"
                "elif argv and argv[0] == 'src/bin/install-pkmn-engine':\n"
                "    expected = Path(os.environ['FAKE_ENGINE_DIR'])\n"
                "    if Path.cwd() != expected:\n"
                "        sys.exit('wrong package working directory')\n"
                "    out = Path('build/lib/pkmn-showdown.node')\n"
                "    out.parent.mkdir(parents=True, exist_ok=True)\n"
                "    out.write_bytes(b'fake native addon')\n"
                "    with open(os.environ['FAKE_BUILD_LOG'], 'a') as f:\n"
                "        f.write('build\\n')\n"
                "else:\n"
                "    sys.exit('unexpected node invocation: ' + repr(argv))\n"
            )
            npm = binaries / "npm"
            npm.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo 10.8.2; '
                "else exit 3; fi\n"
            )
            zig = binaries / "zig"
            zig.write_text("#!/bin/sh\necho 0.16.0\n")
            for binary in (node, npm, zig):
                binary.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "FAKE_ENGINE_DIR": str(engine),
                "FAKE_BUILD_LOG": str(build_log),
            }
            command = ["bash", str(tools_dir / "setup_pkmn_engine.sh")]
            first = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue((host_dir / "pkmn-showdown.node").is_file())
            self.assertEqual(build_log.read_text(), "build\n")
            second = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(build_log.read_text(), "build\n")


if __name__ == "__main__":
    unittest.main()
