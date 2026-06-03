from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "install.sh"


class InstallScriptTests(unittest.TestCase):
    def test_defaults_to_prerelease_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_helper(
                Path(tmp),
                """
                printf '%s\\n' "$INSTALL_PRERELEASE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "1")

    def test_runtime_installer_uses_platform_asset_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = self._run_helper(
                tmp_path,
                """
                MESH_LLM_TEST_UNAME_S=Darwin
                MESH_LLM_TEST_UNAME_M=arm64
                asset_name metal
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "mesh-llm-aarch64-apple-darwin.tar.gz")

    def test_runtime_installer_requires_native_runtime_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            result = self._run_helper(
                tmp_path,
                """
                release_url() {
                    printf 'file:///missing/%s\\n' "$1"
                }
                main
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native runtime manifest was not available", result.stderr)

    def _run_helper(self, tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            source {SCRIPT}
            {body}
            """
        )
        return subprocess.run(
            ["bash", "-c", script],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _write_fake_mesh_llm(self, path: Path, body: str) -> None:
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
