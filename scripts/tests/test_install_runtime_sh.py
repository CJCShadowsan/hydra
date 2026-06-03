from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "install-runtime.sh"


class InstallRuntimeScriptTests(unittest.TestCase):
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

    def test_download_runtime_binary_archive_prefers_platform_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            (assets_dir / platform_asset).write_text("platform\n", encoding="utf-8")
            (assets_dir / "mesh-bundle.tar.gz").write_text("fallback\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                download_runtime_binary_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"asset={platform_asset}", result.stdout)
            self.assertIn(f"archive={tmp_path / platform_asset}", result.stdout)

    def test_download_runtime_binary_archive_falls_back_to_mesh_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            (assets_dir / "mesh-bundle.tar.gz").write_text("fallback\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                download_runtime_binary_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("asset=mesh-bundle.tar.gz", result.stdout)
            self.assertIn(f"archive={tmp_path / 'mesh-bundle.tar.gz'}", result.stdout)
            self.assertIn("Using runtime-enabled mesh bundle", result.stdout)

    def test_install_native_runtime_required_rejects_old_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            (tmp_path / "native-runtimes.json").write_text("{}\n", encoding="utf-8")
            self._write_fake_mesh_llm(
                install_dir / "mesh-llm",
                """
                exit 2
                """,
            )

            result = self._run_helper(
                tmp_path,
                f"""
                INSTALL_DIR={install_dir}
                install_native_runtime_required "{tmp_path}"
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not support native runtime install", result.stderr)

    def test_main_runtime_installs_mesh_bundle_and_required_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            self._write_release_archive(assets_dir / "mesh-bundle.tar.gz")
            (assets_dir / "native-runtimes.json").write_text("{}\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                f"""
                INSTALL_DIR={install_dir}
                MESH_LLM_TEST_UNAME_S=Darwin
                MESH_LLM_TEST_UNAME_M=arm64
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                main
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (install_dir / "runtime-calls.log").read_text(encoding="utf-8")
            self.assertIn("runtime install --manifest ", calls)
            self.assertIn("native-runtimes.json", calls)
            self.assertIn("runtime prune --active-only", calls)
            self.assertIn("Installed mesh-bundle.tar.gz and native runtime", result.stdout)

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

    def _write_release_archive(self, archive_path: Path) -> None:
        script = textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail
            log="$(dirname "$0")/runtime-calls.log"
            if [[ "$*" == "runtime install --help" ]]; then
                exit 0
            fi
            echo "$*" >> "$log"
            exit 0
            """
        )
        data = script.encode("utf-8")
        info = tarfile.TarInfo("mesh-bundle/mesh-llm")
        info.mode = 0o755
        info.size = len(data)

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
