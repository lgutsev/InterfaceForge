from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "launch_scripts"
    / "deepmd_lammps_30_gpu_audit.sbatch"
)


def test_lammps_audit_targets_site_deepmd_module_without_memory_request() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "lammps/29Aug2024-r8.0-deepmd3.0.0-gpu" in text
    assert "#SBATCH --mem" not in text
    assert "type -a -p" in text
    assert "pair_style deepmd" in text


def test_lammps_audit_can_exercise_a_frozen_committee() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DEEPMD_MODELS_FILE" in text
    assert "DEEPMD_SYSTEM" in text
    assert "model_devi.out" in text
    assert 'srun -n 1 "$LMP_BIN"' in text
