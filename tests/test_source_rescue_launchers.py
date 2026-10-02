"""Exercise the spool-copy path used by Slurm, without requiring a scheduler."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ["audit_vasp_sources_single.sbatch", "collect_mapped_leaf_single.sbatch"]


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("mode", ["spool", "explicit_root", "local", "bad_submit_dir"])
def test_launcher_finds_checkout_after_slurm_copy(tmp_path, launcher, mode):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    srun = binaries / "srun"
    srun.write_text('#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
    python = binaries / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], "
        "'pythonpath': os.environ['PYTHONPATH']}))\n"
    )
    python.chmod(0o755)
    policy = tmp_path / "policy.yaml"
    policy.write_text("fixture")
    capture = tmp_path / "capture.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.pop("INTERFACEFORGE_ROOT", None)
    env.update(PATH=f"{binaries}:{env['PATH']}", INTERFACEFORGE_PYTHON=str(python), CAPTURE=str(capture))
    original = ROOT / "launch_scripts" / launcher
    script = original
    if mode != "local":
        spool = tmp_path / "spool"
        spool.mkdir()
        script = spool / "slurm_script"
        shutil.copyfile(original, script)
        env.update(SLURM_JOB_ID="1234", SLURM_SUBMIT_DIR=str(ROOT))
    if mode in {"explicit_root", "bad_submit_dir"}:
        env["SLURM_SUBMIT_DIR"] = str(tmp_path)
    if mode == "explicit_root":
        env["INTERFACEFORGE_ROOT"] = str(ROOT)
    args = ["bash", str(script), str(policy)]
    if launcher.startswith("audit"):
        args.extend([str(tmp_path / "report"), "--logs-only"])
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    if mode == "bad_submit_dir":
        assert result.returncode == 2
        assert "Submit from the InterfaceForge checkout" in result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    recorded = json.loads(capture.read_text())
    assert recorded["pythonpath"].split(":")[0] == str(ROOT / "src")
    assert str(policy) in recorded["args"]
    module = "interfaceforge.source_audit" if launcher.startswith("audit") else "interfaceforge.mapped_collect"
    assert module in recorded["args"]
    if launcher.startswith("audit"):
        assert "--logs-only" in recorded["args"]
        assert recorded["args"][recorded["args"].index("--output") + 1] == str(tmp_path / "report")
