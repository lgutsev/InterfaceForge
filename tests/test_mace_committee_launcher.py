from pathlib import Path

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "launch_scripts"
    / "mace_train_committee.sh"
)


def test_committee_launcher_uses_canonical_reference_keys() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'ENERGY_KEY="${MACE_ENERGY_KEY:-REF_energy}"' in script
    assert 'FORCES_KEY="${MACE_FORCES_KEY:-REF_forces}"' in script
    assert 'MODEL_PREFIX="${MACE_MODEL_PREFIX:-SiN_TiN_TiO_periodic_mace}"' in script


def test_committee_preflight_does_not_require_ase_calculator() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "atoms.info[energy_key]" in script
    assert "atoms.arrays[forces_key]" in script
    assert "atoms.get_potential_energy()" not in script
    assert "atoms.get_forces()" not in script
