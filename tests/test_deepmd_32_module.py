from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE = "deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu"


def test_loni_profiles_preserve_explicit_deepmd_32_module() -> None:
    for path in (
        ROOT / "profiles" / "loni.yaml",
        ROOT / "src" / "interfaceforge" / "templates" / "profile_loni.yaml",
    ):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        job = profile["jobs"]["deepmd_gpu_320"]

        assert job["modules"] == [MODULE]
        assert job["partition"] == "gpu2"
        assert job["gpus"] == 1
        assert "mem" not in job


def test_standalone_preflight_checks_pytorch_gpu_and_has_no_memory_request() -> None:
    script = (
        ROOT / "launch_scripts" / "deepmd_32_gpu_preflight.sbatch"
    ).read_text(encoding="utf-8")

    assert MODULE in script
    assert "torch.cuda.is_available()" in script
    assert '"$DP_BIN" --pt train --help' in script
    assert "#SBATCH --mem" not in script
