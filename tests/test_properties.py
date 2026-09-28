from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import write

from interfaceforge import matcalc_adapter, property_models
from interfaceforge.cli import build_parser
from interfaceforge.cli import main as cli_main
from interfaceforge.errors import ConfigurationError, DependencyError, SafetyError
from interfaceforge.properties import (
    RESULT_FILE,
    Member,
    PropertyConfig,
    require_bulk,
    require_stress,
    run_properties,
    select_ion_relaxation,
    summarize,
    to_jsonable,
)
from interfaceforge.state import sha256_file


class NoStressCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def _fake_backend(bulk_modulus: float = 140.0, *, tensor_shape: tuple[int, int] = (6, 6), converged: bool = True,
                  bprime: float = 4.5, refit_bprime: float | None = None, eos_calls: list | None = None,
                  elastic_calls: list | None = None, shear_softening: float = 0.0,
                  deformed_converged: bool | None = True):
    """MatCalc-shaped results without MatCalc, so the runner is testable anywhere."""

    def relax(atoms, calculator, *, fmax, max_steps, optimizer, member_dir):
        return {
            "record": {"converged": converged, "max_force": 0.001 if converged else 0.5, "n_steps": 3,
                       "hit_max_steps": not converged, "fmax": fmax, "optimizer": optimizer,
                       "energy_per_atom": -3.0, "volume_per_atom": 11.6, "a": 2.54, "b": 2.54, "c": 2.54,
                       "alpha": 60.0, "beta": 60.0, "gamma": 60.0, "max_abs_final_stress": 0.01,
                       "units": {"energy_per_atom": "eV/atom"}},
            "atoms": atoms.copy(),
        }

    def eos(atoms, calculator, **kwargs):
        if eos_calls is not None:
            eos_calls.append(kwargs)
        scale = getattr(calculator, "scale", 1.0)
        fit = {"equilibrium_energy": -3.0, "equilibrium_energy_per_atom": -3.0, "equilibrium_volume": 11.6,
               "equilibrium_volume_per_atom": 11.6, "bulk_modulus": bulk_modulus * scale,
               "bulk_modulus_derivative": bprime, "r2": 0.9999}
        refit = ({"status": "skipped", "reason": "too few points"} if refit_bprime is None
                 else {"status": "ok", **fit, "bulk_modulus": 1.1 * bulk_modulus * scale,
                       "bulk_modulus_derivative": refit_bprime})
        return {**fit, "volumes": np.linspace(10, 13, 5), "near_equilibrium_refit": refit,
                "scan_window": {"max_abs_linear_strain": kwargs.get("max_abs_strain"), "min_volume_ratio": 0.729,
                                "max_volume_ratio": 1.331},
                "warnings": [], "units": {"bulk_modulus": "GPa"}}

    def elasticity(atoms, calculator, **kwargs):
        if elastic_calls is not None:
            elastic_calls.append(kwargs)
        tensor = np.eye(*tensor_shape) * getattr(calculator, "scale", 1.0)
        # Mimic a nonlinear shear response: the diagonal shear C_ii grow with the widest shear strain.
        widest = max(abs(value) for value in kwargs.get("shear_strains", (0.01,)))
        for i in range(3, min(tensor_shape)):
            tensor[i, i] *= 1.0 + shear_softening * (widest - 0.01) / 0.05
        ion_relaxation = kwargs.get("ion_relaxation", "relaxed")
        deformed = None
        if ion_relaxation == "relaxed":
            deformed = {"n_structures": 24, "fmax": kwargs.get("fmax"), "max_steps": kwargs.get("max_steps"),
                        "converged": deformed_converged, "n_converged": 24 if deformed_converged else 20,
                        "max_force": 0.004 if deformed_converged else 0.08,
                        "reason": None if deformed_converged is not None else "not exposed"}
        return {"elastic_tensor": tensor, "tensor_convention": "voigt", "bulk_modulus_vrh": 140.0,
                "ion_relaxation": ion_relaxation, "deformed_relaxations": deformed,
                "shear_modulus_vrh": 60.0 * tensor[3, 3] if min(tensor_shape) > 3 else 60.0,
                "youngs_modulus": 157.0,
                "strains": {"norm_strains": list(kwargs.get("norm_strains", ())),
                            "shear_strains": list(kwargs.get("shear_strains", ()))},
                "warnings": [], "units": {}}

    def phonon(atoms, calculator, **_):
        return {"frequencies": np.ones((4, 3)), "min_frequency": 0.0, "dynamically_stable": True,
                "thermal_properties": {"temperatures": np.arange(3.0), "heat_capacity": np.ones(3)}, "units": {}}

    return SimpleNamespace(relax=relax, eos=eos, elasticity=elasticity, phonon=phonon)


def _scaled_emt(scale: float):
    def make():
        calc = EMT()
        calc.scale = scale
        return calc

    return make


def _cu() -> Atoms:
    return bulk("Cu", "fcc", a=3.6)


def _hcp_cu() -> Atoms:
    # Two atoms on -6m2 sites (no inversion centre): in-plane strain moves them, so C11/C12/C66 relax.
    return bulk("Cu", "hcp", a=2.55, c=4.16)


# ---------------------------------------------------------------- dependency


def test_missing_matcalc_raises_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matcalc" or name.startswith("matcalc."):
            raise ImportError("no matcalc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DependencyError, match=r"\.\[properties\].*|Python >= 3.11"):
        matcalc_adapter.require_matcalc()


def test_missing_matcalc_is_reported_by_runner(monkeypatch, tmp_path):
    def missing():
        raise DependencyError(matcalc_adapter.INSTALL_HINT)

    monkeypatch.setattr(matcalc_adapter, "require_matcalc", missing)
    with pytest.raises(DependencyError, match="Python >= 3.11"):
        run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), tmp_path / "out", engine="emt")
    assert not (tmp_path / "out").exists()


def test_importing_cli_does_not_import_matcalc_or_backends():
    import subprocess
    import sys

    heavy = "('matcalc', 'phonopy', 'mace', 'deepmd', 'torch')"
    code = f"import sys, interfaceforge.cli; print(sorted(m for m in {heavy} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------------- CLI


def test_cli_parses_repeatable_models_and_properties():
    args = build_parser().parse_args(
        ["properties", "run", "TiN.vasp", "--engine", "mace", "--model", "seed_11.model", "--model", "seed_23.model",
         "--property", "eos", "--property", "elasticity", "--output", "out", "--dtype", "float64", "--fmax", "0.02"]
    )
    assert args.model == ["seed_11.model", "seed_23.model"]
    assert args.property == ["eos", "elasticity"]
    assert (args.engine, args.dtype, args.fmax, args.max_steps, args.device) == ("mace", "float64", 0.02, 500, "cpu")


def test_invalid_property_name_is_rejected():
    with pytest.raises(ConfigurationError, match="Unknown property 'bandgap'"):
        PropertyConfig(("eos", "bandgap"))
    with pytest.raises(ConfigurationError, match="at least one"):
        PropertyConfig(())


def test_eos_window_config_validation():
    config = PropertyConfig(("eos",))
    assert (config.eos_max_strain, config.eos_points, config.eos_refit_window) == (0.1, 11, 0.05)
    PropertyConfig(("eos",), eos_max_strain=0.2, eos_points=5, eos_refit_window=0.2)
    for strain in (0.0, -0.05, 0.21, float("nan")):
        with pytest.raises(ConfigurationError, match="--eos-max-strain"):
            PropertyConfig(("eos",), eos_max_strain=strain)
    for points in (3, 4, 10, 11.0, True):
        with pytest.raises(ConfigurationError, match="--eos-points"):
            PropertyConfig(("eos",), eos_points=points)
    for window in (0.0, 0.3):
        with pytest.raises(ConfigurationError, match="--eos-refit-window"):
            PropertyConfig(("eos",), eos_refit_window=window)


def test_cli_parses_eos_window_options():
    base = ["properties", "run", "Si3N4.vasp", "--engine", "mace", "--model", "m.model", "--property", "eos",
            "--output", "out"]
    args = build_parser().parse_args(base)
    assert (args.eos_max_strain, args.eos_points, args.eos_refit_window) == (0.1, 11, 0.05)
    args = build_parser().parse_args([*base, "--eos-max-strain", "0.05", "--eos-points", "21",
                                      "--eos-refit-window", "0.04"])
    assert (args.eos_max_strain, args.eos_points, args.eos_refit_window) == (0.05, 21, 0.04)


def test_elastic_strain_config_validation():
    config = PropertyConfig(("elasticity",))
    assert config.elastic_norm_strains == (-0.01, -0.005, 0.005, 0.01)
    assert config.elastic_shear_strains == (-0.01, -0.005, 0.005, 0.01)
    assert config.elastic_strain_check is False
    custom = PropertyConfig(("elasticity",), elastic_norm_strains=[0.1, -0.1],
                            elastic_shear_strains=np.array([1e-3, 2e-3]))
    assert custom.elastic_norm_strains == (0.1, -0.1) and custom.elastic_shear_strains == (0.001, 0.002)
    assert all(type(value) is float for value in custom.elastic_shear_strains)
    bad = ((0.01,), (), (0.0, 0.01), (0.01, 0.11), (-0.2, 0.01), (0.01, float("nan")), (0.01, float("inf")),
           (True, 0.01), "0.01 0.02", 0.01, ("a", "b"))
    for option, name in (("--elastic-norm-strains", "elastic_norm_strains"),
                         ("--elastic-shear-strains", "elastic_shear_strains")):
        for strains in bad:
            with pytest.raises(ConfigurationError, match=option):
                PropertyConfig(("elasticity",), **{name: strains})
    with pytest.raises(ConfigurationError, match="--elastic-strain-check needs --property elasticity"):
        PropertyConfig(("eos",), elastic_strain_check=True)


def test_elastic_check_window_is_the_other_window():
    matcalc = ((-0.01, -0.005, 0.005, 0.01), (-0.06, -0.03, 0.03, 0.06))
    small = ((-0.01, -0.005, 0.005, 0.01), (-0.01, -0.005, 0.005, 0.01))
    assert PropertyConfig(("elasticity",)).elastic_check_strains == matcalc
    assert PropertyConfig(("elasticity",), elastic_shear_strains=matcalc[1]).elastic_check_strains == small
    assert PropertyConfig(("elasticity",), elastic_shear_strains=(-0.02, 0.02)).elastic_check_strains == matcalc


def test_cli_parses_elastic_strain_options():
    base = ["properties", "run", "TiN.vasp", "--engine", "mace", "--model", "m.model", "--property", "elasticity",
            "--output", "out"]
    args = build_parser().parse_args(base)
    assert args.elastic_norm_strains == [-0.01, -0.005, 0.005, 0.01]
    assert args.elastic_shear_strains == [-0.01, -0.005, 0.005, 0.01]
    assert args.elastic_strain_check is False
    args = build_parser().parse_args([*base, "--elastic-shear-strains", "-0.06", "-0.03", "0.03", "0.06",
                                      "--elastic-norm-strains", "-0.005", "0.005", "--elastic-strain-check"])
    assert args.elastic_shear_strains == [-0.06, -0.03, 0.03, 0.06]
    assert args.elastic_norm_strains == [-0.005, 0.005]
    assert args.elastic_strain_check is True


def test_elastic_ion_relaxation_config_validation():
    config = PropertyConfig(("elasticity",))
    assert config.elastic_ion_relaxation == "auto" and config.elastic_fmax == 0.005
    for choice in ("auto", "relaxed", "clamped"):
        assert PropertyConfig(("elasticity",), elastic_ion_relaxation=choice).elastic_ion_relaxation == choice
    with pytest.raises(ConfigurationError, match="--elastic-ion-relaxation must be one of auto, relaxed, clamped"):
        PropertyConfig(("elasticity",), elastic_ion_relaxation="frozen")
    for bad in (0.0, -0.001, float("nan"), float("inf"), True, "0.005"):
        with pytest.raises(ConfigurationError, match="--elastic-fmax"):
            PropertyConfig(("elasticity",), elastic_fmax=bad)


def test_cli_parses_elastic_ion_relaxation_options():
    base = ["properties", "run", "TiN.vasp", "--engine", "mace", "--model", "m.model", "--property", "elasticity",
            "--output", "out"]
    args = build_parser().parse_args(base)
    assert (args.elastic_ion_relaxation, args.elastic_fmax) == ("auto", 0.005)
    args = build_parser().parse_args([*base, "--elastic-ion-relaxation", "clamped", "--elastic-fmax", "0.002"])
    assert (args.elastic_ion_relaxation, args.elastic_fmax) == ("clamped", 0.002)
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--elastic-ion-relaxation", "frozen"])


def test_auto_ion_relaxation_follows_site_inversion_symmetry():
    pytest.importorskip("spglib")
    clamped = {"fcc Cu": _cu(), "rocksalt TiN": bulk("TiN", "rocksalt", a=4.24),
               "rocksalt TiN supercell": bulk("TiN", "rocksalt", a=4.24, cubic=True).repeat(2),
               "CsCl": bulk("CsCl", "cesiumchloride", a=4.1)}
    relaxed = {"hcp Cu": _hcp_cu(), "diamond Si": bulk("Si", "diamond", a=5.43),
               "wurtzite ZnO": bulk("ZnO", "wurtzite", a=3.25, c=5.2),
               "fluorite CaF2": bulk("CaF2", "fluorite", a=5.46)}
    for name, atoms in clamped.items():
        selection = select_ion_relaxation("auto", atoms)
        assert selection["ion_relaxation"] == "clamped", name
        assert selection["non_centrosymmetric_sites"] == []
    for name, atoms in relaxed.items():
        assert select_ion_relaxation("auto", atoms)["ion_relaxation"] == "relaxed", name
    # Fluorite: Ca sits on an inversion centre, the two F do not.
    assert select_ion_relaxation("auto", bulk("CaF2", "fluorite", a=5.46))["non_centrosymmetric_sites"] == [1, 2]
    # A rattled rock salt has lost its inversion centres, so it is relaxed-ion.
    rattled = bulk("TiN", "rocksalt", a=4.24, cubic=True)
    rattled.rattle(0.01, seed=1)
    assert select_ion_relaxation("auto", rattled)["ion_relaxation"] == "relaxed"


def test_explicit_ion_relaxation_overrides_symmetry():
    assert select_ion_relaxation("relaxed", _cu())["ion_relaxation"] == "relaxed"
    selection = select_ion_relaxation("clamped", _hcp_cu())
    assert selection["ion_relaxation"] == "clamped" and selection["reason"] == "explicit --elastic-ion-relaxation"


def test_auto_ion_relaxation_defaults_to_relaxed_without_spglib(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "spglib":
            raise ImportError("no spglib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    selection = select_ion_relaxation("auto", _cu())
    assert selection["ion_relaxation"] == "relaxed" and "could not be determined" in selection["reason"]


def test_cli_invalid_elastic_strains_exit_with_error(tmp_path, capsys):
    model = tmp_path / "m.model"
    model.write_bytes(b"x")
    code = cli_main(["properties", "run", "x.vasp", "--engine", "mace", "--model", str(model),
                     "--property", "elasticity", "--elastic-shear-strains", "0.01", "0.0",
                     "--output", str(tmp_path / "o")])
    assert code == 2
    assert "--elastic-shear-strains" in capsys.readouterr().err


def test_cli_invalid_eos_points_exits_with_error(tmp_path, capsys):
    model = tmp_path / "m.model"
    model.write_bytes(b"x")
    code = cli_main(["properties", "run", "x.vasp", "--engine", "mace", "--model", str(model), "--property", "eos",
                     "--eos-points", "10", "--output", str(tmp_path / "o")])
    assert code == 2
    assert "--eos-points" in capsys.readouterr().err


def test_cli_invalid_property_exits_with_error(tmp_path, capsys):
    model = tmp_path / "m.model"
    model.write_bytes(b"x")
    code = cli_main(["properties", "run", "x.vasp", "--engine", "mace", "--model", str(model),
                     "--property", "hardness", "--output", str(tmp_path / "o")])
    assert code == 2
    assert "Unknown property" in capsys.readouterr().err


def test_cli_missing_model(tmp_path, capsys):
    code = cli_main(["properties", "run", "x.vasp", "--engine", "mace", "--model", str(tmp_path / "nope.model"),
                     "--property", "eos", "--output", str(tmp_path / "o")])
    assert code == 2
    assert "Model not found" in capsys.readouterr().err


def test_deepmd_rejects_dtype_and_mace_defaults_to_float64():
    assert property_models.resolve_dtype("mace", None) == "float64"
    assert property_models.resolve_dtype("mace", "float32") == "float32"
    assert property_models.resolve_dtype("deepmd", None) is None
    with pytest.raises(ConfigurationError, match="frozen DeePMD"):
        property_models.resolve_dtype("deepmd", "float64")


def test_model_seed_and_labels_recovered(tmp_path):
    paths = []
    for seed in (11, 23):
        path = tmp_path / f"seed_{seed}" / "model.model"
        path.parent.mkdir()
        path.write_bytes(str(seed).encode())
        paths.append(str(path))
    specs = property_models.models_from_paths(paths, "mace")
    assert [spec.seed for spec in specs] == [11, 23]
    assert [spec.label for spec in specs] == ["member_000_seed_11", "member_001_seed_23"]
    with pytest.raises(SafetyError, match="Duplicate"):
        property_models.models_from_paths([paths[0], paths[0]], "mace")


def test_committee_bundle_is_verified_and_engine_checked(tmp_path):
    from interfaceforge.committee import collect_committee

    source = tmp_path / "runs"
    for seed in (11, 23):
        run = source / f"seed_{seed}" / "mace_model"
        run.mkdir(parents=True)
        (run / f"seed_{seed}_stagetwo.model").write_bytes(f"model-{seed}".encode())
    collect_committee(source, tmp_path / "committee", engine="mace", expected_members=2)
    specs = property_models.models_from_committee(str(tmp_path / "committee"), "mace")
    assert [spec.seed for spec in specs] == [11, 23]
    assert all(spec.path.is_file() and spec.source.startswith("committee:") for spec in specs)
    with pytest.raises(ConfigurationError, match="mace bundle"):
        property_models.models_from_committee(str(tmp_path / "committee"), "deepmd")


# ------------------------------------------------------------------- guards


def test_bulk_guard_accepts_bulk():
    assert require_bulk(_cu())["fully_periodic"]


def test_bulk_guard_rejects_non_periodic_molecule():
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    with pytest.raises(SafetyError, match="3D-periodic"):
        require_bulk(atoms)


def test_bulk_guard_rejects_slab_even_when_marked_fully_periodic():
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    slab.pbc = True
    with pytest.raises(SafetyError, match="slab with vacuum"):
        require_bulk(slab)
    slab.pbc = (True, True, False)
    with pytest.raises(SafetyError, match="3D-periodic"):
        require_bulk(slab)


def test_runner_rejects_slab_before_touching_output(tmp_path):
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=8.0, periodic=True)
    with pytest.raises(SafetyError):
        run_properties(slab, [Member("m0", EMT)], PropertyConfig(("eos",)), tmp_path / "o", engine="emt",
                       backend=_fake_backend())
    assert not (tmp_path / "o").exists()


def test_stress_check():
    require_stress(_cu(), EMT())
    with pytest.raises(SafetyError, match="does not implement stress"):
        require_stress(_cu(), NoStressCalculator())


def test_stress_unsupported_member_fails_clearly(tmp_path):
    payload = run_properties(_cu(), [Member("m0", NoStressCalculator)], PropertyConfig(("elasticity",)),
                             tmp_path / "o", engine="emt", backend=_fake_backend())
    model = payload["models"][0]
    assert model["status"] == "failed" and model["failed_property"] == "stress_check"
    assert "stress" in model["error"] and model["results"] == {}
    assert payload["summary"]["properties"]["elasticity"]["n_members"] == 0


# ------------------------------------------------------------------- output


def test_output_overwrite_protection(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    args = (_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), out)
    with pytest.raises(SafetyError, match="not empty"):
        run_properties(*args, engine="emt", backend=_fake_backend())
    with pytest.raises(SafetyError, match="not a prior property result"):
        run_properties(*args, engine="emt", backend=_fake_backend(), force=True)
    assert (out / "notes.txt").read_text() == "keep me"

    fresh = tmp_path / "fresh"
    run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), fresh, engine="emt", backend=_fake_backend())
    with pytest.raises(SafetyError, match="not empty"):
        run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), fresh, engine="emt",
                       backend=_fake_backend())
    run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), fresh, engine="emt",
                   backend=_fake_backend(), force=True)
    assert json.loads((fresh / RESULT_FILE).read_text())["schema_version"] == 1
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_provenance_and_model_hashing(tmp_path):
    structure = tmp_path / "Cu.vasp"
    write(structure, _cu(), format="vasp")
    model = tmp_path / "seed_7.model"
    model.write_bytes(b"weights")
    before = (sha256_file(structure), sha256_file(model))
    member = Member("member_000_seed_7", EMT, hashed_path=model,
                    record={"model_path": str(model), "model_sha256": sha256_file(model), "seed": 7})
    payload = run_properties(_cu(), [member], PropertyConfig(("relax", "eos")), tmp_path / "o", engine="mace",
                             structure_source=structure, backend=_fake_backend())
    on_disk = json.loads((tmp_path / "o" / RESULT_FILE).read_text())
    assert on_disk["structure"]["sha256"] == before[0]
    assert on_disk["structure"]["formula"] == "Cu" and on_disk["structure"]["pbc"] == [True, True, True]
    assert len(on_disk["structure"]["cell"]) == 3 and on_disk["structure"]["n_atoms"] == 1
    assert on_disk["models"][0]["model_sha256"] == before[1]
    assert on_disk["models"][0]["package_versions"]["ase"]
    assert {"interfaceforge", "matcalc", "ase", "mace-torch"} <= set(on_disk["packages"])
    assert on_disk["inputs_unchanged"] is True
    assert (sha256_file(structure), sha256_file(model)) == before
    assert payload["models"][0]["calculator_class"].endswith("EMT")


# ---------------------------------------------------------------- summaries


def test_one_member_result(tmp_path):
    payload = run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), tmp_path / "o", engine="emt",
                             backend=_fake_backend())
    stats = payload["summary"]["properties"]["eos"]["scalars"]["bulk_modulus"]
    assert payload["summary"]["success_fraction"] == "1/1"
    assert stats["mean"] == 140.0 and stats["std"] is None and stats["n"] == 1


def test_multi_member_aggregation_is_per_member(tmp_path):
    members = [Member(f"m{i}", _scaled_emt(scale)) for i, scale in enumerate((0.9, 1.0, 1.1))]
    payload = run_properties(_cu(), members, PropertyConfig(("eos", "elasticity", "phonon")), tmp_path / "o",
                             engine="emt", backend=_fake_backend(100.0))
    props = payload["summary"]["properties"]
    stats = props["eos"]["scalars"]["bulk_modulus"]
    assert [m["results"]["eos"]["bulk_modulus"] for m in payload["models"]] == pytest.approx([90.0, 100.0, 110.0])
    assert stats["mean"] == pytest.approx(100.0) and stats["std"] == pytest.approx(10.0)
    assert (stats["min"], stats["max"], stats["n"]) == pytest.approx((90.0, 110.0, 3))
    assert props["elasticity"]["elastic_tensor"]["aggregated"]
    assert np.allclose(props["elasticity"]["elastic_tensor"]["mean"], np.eye(6))
    assert props["phonon"]["frequencies"]["aggregated"]
    assert not [key for key in props["phonon"] if "score" in key]


def test_mismatched_arrays_are_not_aggregated():
    models = [
        {"label": "a", "status": "ok", "results": {"elasticity": {"elastic_tensor": np.eye(6), "tensor_convention": "v",
                                                                  "bulk_modulus_vrh": 1, "shear_modulus_vrh": 1,
                                                                  "youngs_modulus": 1}}},
        {"label": "b", "status": "ok", "results": {"elasticity": {"elastic_tensor": np.eye(3), "tensor_convention": "v",
                                                                  "bulk_modulus_vrh": 1, "shear_modulus_vrh": 1,
                                                                  "youngs_modulus": 1}}},
    ]
    entry = summarize(models, ["elasticity"])["properties"]["elasticity"]["elastic_tensor"]
    assert entry["aggregated"] is False and "shapes differ" in entry["reason"]


def test_one_failing_member_is_reported_not_hidden(tmp_path):
    def broken():
        raise RuntimeError("checkpoint is corrupt")

    members = [Member("m0", EMT), Member("m1", broken), Member("m2", EMT), Member("m3", EMT)]
    payload = run_properties(_cu(), members, PropertyConfig(("eos",)), tmp_path / "o", engine="emt",
                             backend=_fake_backend())
    summary = payload["summary"]
    assert summary["success_fraction"] == "3/4" and summary["n_failed"] == 1
    assert summary["failed_members"][0]["label"] == "m1"
    assert "checkpoint is corrupt" in summary["failed_members"][0]["error"]
    assert summary["properties"]["eos"]["scalars"]["bulk_modulus"]["n"] == 3


def test_eos_window_is_plumbed_to_backend_and_recorded(tmp_path):
    calls: list = []
    config = PropertyConfig(("eos",), eos_max_strain=0.05, eos_points=21, eos_refit_window=0.04)
    payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(eos_calls=calls, refit_bprime=4.0))
    assert calls == [{"fmax": 0.01, "max_steps": 500, "optimizer": "FIRE", "max_abs_strain": 0.05, "n_points": 21,
                      "refit_volume_window": 0.04}]
    on_disk = json.loads((tmp_path / "o" / RESULT_FILE).read_text())
    assert (on_disk["settings"]["eos_max_strain"], on_disk["settings"]["eos_points"]) == (0.05, 21)
    model = on_disk["models"][0]
    assert model["status"] == "ok" and model["warnings"] == []
    eos = model["results"]["eos"]
    assert eos["bulk_modulus_derivative_plausible"] is True
    assert eos["near_equilibrium_refit"]["bulk_modulus_derivative_plausible"] is True
    refit = payload["summary"]["properties"]["eos"]["near_equilibrium_refit"]
    assert refit["n_members"] == 1 and refit["scalars"]["bulk_modulus"]["mean"] == pytest.approx(154.0)


def test_elastic_strains_are_plumbed_to_backend_and_recorded(tmp_path):
    calls: list = []
    config = PropertyConfig(("elasticity",), elastic_norm_strains=(-0.005, 0.005),
                            elastic_shear_strains=(-0.02, 0.02), elastic_ion_relaxation="clamped")
    payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(elastic_calls=calls))
    assert calls == [{"fmax": 0.005, "norm_strains": (-0.005, 0.005), "shear_strains": (-0.02, 0.02),
                      "ion_relaxation": "clamped", "max_steps": 500, "optimizer": "FIRE"}]
    on_disk = json.loads((tmp_path / "o" / RESULT_FILE).read_text())
    settings = on_disk["settings"]
    assert settings["elastic_norm_strains"] == [-0.005, 0.005] and settings["elastic_shear_strains"] == [-0.02, 0.02]
    assert settings["elastic_strain_check"] is False
    elastic = on_disk["models"][0]["results"]["elasticity"]
    assert elastic["strains"]["shear_strains"] == [-0.02, 0.02] and "strain_check" not in elastic
    assert "strain_check" not in payload["summary"]["properties"]["elasticity"]


def test_elastic_strain_check_passes_for_a_linear_response(tmp_path):
    calls: list = []
    config = PropertyConfig(("elasticity",), elastic_strain_check=True)
    payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(elastic_calls=calls))
    assert [call["shear_strains"] for call in calls] == [(-0.01, -0.005, 0.005, 0.01), (-0.06, -0.03, 0.03, 0.06)]
    model = payload["models"][0]
    assert model["status"] == "ok" and model["warnings"] == []
    check = model["results"]["elasticity"]["strain_check"]
    assert check["flagged"] is False and check["flagged_components"] == []
    assert check["max_diagonal_relative_change"] == pytest.approx(0.0)
    assert check["reference_strains"]["shear_strains"] == [-0.06, -0.03, 0.03, 0.06]
    summary = payload["summary"]["properties"]["elasticity"]["strain_check"]
    assert summary["n_members"] == 1 and summary["flagged_members"] == 0


def test_elastic_strain_check_flags_nonlinear_shear(tmp_path):
    members = [Member("m0", EMT), Member("m1", EMT)]
    config = PropertyConfig(("elasticity",), elastic_strain_check=True)
    payload = run_properties(_cu(), members, config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(shear_softening=0.15))
    model = payload["models"][0]
    assert model["status"] == "ok"
    check = model["results"]["elasticity"]["strain_check"]
    assert check["flagged"] is True and check["flagged_components"] == ["C44", "C55", "C66"]
    assert list(check["diagonal_relative_change"][:3]) == pytest.approx([0.0, 0.0, 0.0])
    assert check["diagonal_relative_change"][3] == pytest.approx(0.15)
    assert len(model["warnings"]) == 1 and "C44 1 -> 1.15 GPa" in model["warnings"][0]
    assert model["results"]["elasticity"]["warnings"] == model["warnings"]
    assert payload["summary"]["properties"]["elasticity"]["strain_check"]["flagged_members"] == 2
    json.loads((tmp_path / "o" / RESULT_FILE).read_text())  # the check block serializes


def test_elastic_strain_check_below_tolerance_is_not_flagged(tmp_path):
    payload = run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("elasticity",), elastic_strain_check=True),
                             tmp_path / "o", engine="emt", backend=_fake_backend(shear_softening=0.09))
    check = payload["models"][0]["results"]["elasticity"]["strain_check"]
    assert check["flagged"] is False and check["max_diagonal_relative_change"] == pytest.approx(0.09)


def test_auto_ion_relaxation_is_plumbed_to_backend_and_recorded(tmp_path):
    pytest.importorskip("spglib")
    calls: list = []
    config = PropertyConfig(("elasticity",), max_steps=300, optimizer="BFGS", elastic_strain_check=True)
    payload = run_properties(_hcp_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(elastic_calls=calls))
    # The strain-check refit uses the same ion treatment and deformed-structure relaxation settings.
    assert [(call["ion_relaxation"], call["fmax"], call["max_steps"], call["optimizer"]) for call in calls] == [
        ("relaxed", 0.005, 300, "BFGS"), ("relaxed", 0.005, 300, "BFGS")]
    on_disk = json.loads((tmp_path / "o" / RESULT_FILE).read_text())
    assert on_disk["settings"]["elastic_ion_relaxation"] == "auto" and on_disk["settings"]["elastic_fmax"] == 0.005
    model = on_disk["models"][0]
    assert model["status"] == "ok" and model["warnings"] == []
    elastic = model["results"]["elasticity"]
    assert elastic["ion_relaxation"] == "relaxed"
    assert elastic["ion_relaxation_selection"]["requested"] == "auto"
    assert elastic["ion_relaxation_selection"]["non_centrosymmetric_sites"] == [0, 1]
    assert elastic["deformed_relaxations"]["converged"] is True
    summary = payload["summary"]["properties"]["elasticity"]
    assert summary["ion_relaxation"] == {"relaxed": 1}
    assert summary["deformed_relaxations"] == {"n_members": 1, "unconverged_members": 0, "unchecked_members": 0}


def test_auto_ion_relaxation_clamps_a_centrosymmetric_cell(tmp_path):
    pytest.importorskip("spglib")
    calls: list = []
    payload = run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("elasticity",)), tmp_path / "o",
                             engine="emt", backend=_fake_backend(elastic_calls=calls))
    assert [call["ion_relaxation"] for call in calls] == ["clamped"]
    elastic = payload["models"][0]["results"]["elasticity"]
    assert elastic["ion_relaxation"] == "clamped" and elastic["deformed_relaxations"] is None
    summary = payload["summary"]["properties"]["elasticity"]
    assert summary["ion_relaxation"] == {"clamped": 1} and "deformed_relaxations" not in summary


def test_unconverged_deformed_relaxations_are_flagged_not_failed(tmp_path):
    members = [Member("m0", EMT), Member("m1", EMT)]
    config = PropertyConfig(("elasticity",), elastic_ion_relaxation="relaxed")
    payload = run_properties(_cu(), members, config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(deformed_converged=False))
    model = payload["models"][0]
    assert model["status"] == "ok"
    assert len(model["warnings"]) == 1 and "4 of 24 deformed structures did not relax" in model["warnings"][0]
    assert model["results"]["elasticity"]["warnings"] == model["warnings"]
    summary = payload["summary"]["properties"]["elasticity"]["deformed_relaxations"]
    assert summary == {"n_members": 2, "unconverged_members": 2, "unchecked_members": 0}


def test_unchecked_deformed_relaxations_are_flagged(tmp_path):
    config = PropertyConfig(("elasticity",), elastic_ion_relaxation="relaxed")
    payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt",
                             backend=_fake_backend(deformed_converged=None))
    assert "convergence is unchecked (not exposed)" in payload["models"][0]["warnings"][0]
    assert payload["summary"]["properties"]["elasticity"]["deformed_relaxations"]["unchecked_members"] == 1


def test_mixed_ion_relaxation_is_not_aggregated():
    def model(label, treatment):
        return {"label": label, "status": "ok", "results": {"elasticity": {
            "elastic_tensor": np.eye(6), "tensor_convention": "v", "ion_relaxation": treatment,
            "bulk_modulus_vrh": 1.0, "shear_modulus_vrh": 1.0, "youngs_modulus": 1.0}}}

    entry = summarize([model("a", "relaxed"), model("b", "clamped")], ["elasticity"])["properties"]["elasticity"]
    assert entry["ion_relaxation"] == {"clamped": 1, "relaxed": 1}
    assert entry["elastic_tensor"]["aggregated"] is False
    assert "ion relaxation differs" in entry["elastic_tensor"]["reason"]


def test_deformed_relaxation_record():
    ok = matcalc_adapter._deformed_relaxation_record(
        [{"is_converged": True, "max_force": 0.001}, {"is_converged": False, "max_force": 0.02}], 2,
        fmax=0.005, max_steps=10, optimizer="FIRE")
    assert ok["convergence_checked"] and ok["converged"] is False and ok["n_converged"] == 1
    assert ok["max_force"] == 0.02 and ok["per_structure_max_force"] == [0.001, 0.02]
    missing = matcalc_adapter._deformed_relaxation_record(None, 24, fmax=0.005, max_steps=10, optimizer="FIRE")
    assert missing["convergence_checked"] is False and missing["converged"] is None
    short = matcalc_adapter._deformed_relaxation_record([], 24, fmax=0.005, max_steps=10, optimizer="FIRE")
    assert short["converged"] is None and "recorded 0 relaxations for 24" in short["reason"]


def test_implausible_bprime_is_flagged_not_failed(tmp_path):
    members = [Member("m0", EMT), Member("m1", EMT)]
    payload = run_properties(_cu(), members, PropertyConfig(("eos",)), tmp_path / "o", engine="emt",
                             backend=_fake_backend(bprime=1.3, refit_bprime=8.5))
    model = payload["models"][0]
    assert model["status"] == "ok"
    assert len(model["warnings"]) == 2
    assert "full scan B'=1.3" in model["warnings"][0] and "0.729-1.331" in model["warnings"][0]
    assert "near-equilibrium refit B'=8.5" in model["warnings"][1]
    eos = model["results"]["eos"]
    assert eos["bulk_modulus_derivative_plausible"] is False and eos["warnings"] == model["warnings"]
    summary = payload["summary"]["properties"]["eos"]
    assert summary["implausible_bulk_modulus_derivative_members"] == 2
    assert summary["near_equilibrium_refit"]["implausible_bulk_modulus_derivative_members"] == 2


def test_skipped_refit_is_not_summarized(tmp_path):
    payload = run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("eos",)), tmp_path / "o", engine="emt",
                             backend=_fake_backend())
    assert payload["models"][0]["warnings"] == []
    refit = payload["summary"]["properties"]["eos"]["near_equilibrium_refit"]
    assert refit == {"n_members": 0, "scalars": {}, "implausible_bulk_modulus_derivative_members": 0}


def test_near_equilibrium_refit_selects_points_inside_window():
    pytest.importorskip("pymatgen")
    v0, b0, b1 = 100.0, 1.3, 4.0  # b0 in eV/A^3 (~208 GPa)

    def birch_murnaghan(volume):
        x = (v0 / volume) ** (2 / 3) - 1
        return -10.0 + 9 * v0 * b0 / 16 * (x**3 * b1 + x**2 * (6 - 4 * (v0 / volume) ** (2 / 3)))

    volumes = v0 * (1 + np.linspace(-0.05, 0.05, 21)) ** 3
    energies = birch_murnaghan(volumes)
    refit = matcalc_adapter._near_equilibrium_refit(volumes, energies, v0, 0.05, n_atoms=10)
    assert refit["status"] == "ok" and refit["n_points"] == 7
    assert refit["bulk_modulus"] == pytest.approx(b0 * matcalc_adapter.EV_PER_A3_TO_GPA, rel=1e-3)
    assert refit["bulk_modulus_derivative"] == pytest.approx(b1, rel=1e-2)
    coarse = v0 * (1 + np.linspace(-0.1, 0.1, 11)) ** 3
    skipped = matcalc_adapter._near_equilibrium_refit(coarse, birch_murnaghan(coarse), v0, 0.05, n_atoms=10)
    assert skipped["status"] == "skipped" and skipped["n_points"] == 1 and "need 5" in skipped["reason"]


def test_unconverged_relaxation_blocks_downstream_properties(tmp_path):
    payload = run_properties(_cu(), [Member("m0", EMT)], PropertyConfig(("relax", "eos", "elasticity")),
                             tmp_path / "o", engine="emt", backend=_fake_backend(converged=False))
    model = payload["models"][0]
    assert model["status"] == "failed" and model["failed_property"] == "relax"
    assert set(model["results"]) == {"relax"} and model["results"]["relax"]["converged"] is False


def test_json_serialization_of_numpy_and_pymatgen_like_values():
    class Obj:
        def as_dict(self):
            return {"x": np.float32(1.5)}

    value = {"a": np.arange(3), "b": np.float64("nan"), "c": np.int64(2), "d": np.bool_(True), "e": Obj(),
             "f": Path("p"), "g": (1, 2)}
    text = json.dumps(to_jsonable(value), allow_nan=False)
    assert json.loads(text) == {"a": [0, 1, 2], "b": None, "c": 2, "d": True, "e": {"x": 1.5}, "f": "p", "g": [1, 2]}
    with pytest.raises(TypeError):
        to_jsonable(object())


# ------------------------------------------------------ MatCalc integration


def test_matcalc_emt_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matcalc")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    structure = tmp_path / "Cu.vasp"
    write(structure, _cu(), format="vasp")
    config = PropertyConfig(("relax", "eos", "elasticity", "phonon"), fmax=0.005, phonon_min_length=6.0)
    try:
        payload = run_properties(_cu(), [Member("m0", EMT), Member("m1", EMT)], config, tmp_path / "o",
                                 engine="emt", structure_source=structure)
    except DependencyError:
        pytest.skip("matcalc import failed")
    models = payload["models"]
    if any("phonors" in str(model.get("error")) for model in models):
        pytest.skip("installed phonopy/phonors pair is incompatible (phonopy 4.5 needs phonors<0.5)")
    assert [model["status"] for model in models] == ["ok", "ok"], [model.get("error") for model in models]
    relax = models[0]["results"]["relax"]
    assert relax["converged"] and relax["n_steps"] is not None and relax["max_force"] <= 0.005
    eos = models[0]["results"]["eos"]
    assert 100 < eos["bulk_modulus"] < 170 and eos["units"]["bulk_modulus"] == "GPa"
    window = eos["scan_window"]
    assert window["max_abs_linear_strain"] == 0.1 and window["n_points_requested"] == 11 == eos["n_points"]
    assert window["min_volume_ratio"] == pytest.approx(0.9**3, rel=1e-3)
    assert window["max_volume_ratio"] == pytest.approx(1.1**3, rel=1e-3)
    assert eos["near_equilibrium_refit"]["status"] == "skipped"  # the default grid has 1 point within +/-5 %
    elastic = models[0]["results"]["elasticity"]
    assert np.asarray(elastic["elastic_tensor"]).shape == (6, 6)
    assert elastic["strains"]["norm_strains"] == [-0.01, -0.005, 0.005, 0.01]
    assert elastic["strains"]["shear_strains"] == [-0.01, -0.005, 0.005, 0.01]
    assert elastic["ion_relaxation"] == "clamped" and elastic["deformed_relaxations"] is None  # fcc: 1-atom cell
    assert models[0]["results"]["phonon"]["dynamically_stable"]
    out = tmp_path / "o"
    assert json.loads((out / RESULT_FILE).read_text())["inputs_unchanged"] is True
    for name in ("relax.traj", "relaxed.vasp", "phonon.yaml"):
        assert (out / "members" / "m0" / name).is_file()
    assert list(cwd.iterdir()) == []  # MatCalc's default phonon.yaml-in-CWD is redirected


def test_matcalc_emt_narrow_eos_window(tmp_path):
    pytest.importorskip("matcalc")
    config = PropertyConfig(("eos",), fmax=0.005, eos_max_strain=0.05, eos_points=21)
    try:
        payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt")
    except DependencyError:
        pytest.skip("matcalc import failed")
    model = payload["models"][0]
    assert model["status"] == "ok", model.get("error")
    eos = model["results"]["eos"]
    assert eos["n_points"] == 21
    assert eos["scan_window"]["min_volume_ratio"] == pytest.approx(0.95**3, rel=1e-3)
    assert eos["scan_window"]["max_volume_ratio"] == pytest.approx(1.05**3, rel=1e-3)
    refit = eos["near_equilibrium_refit"]
    assert refit["status"] == "ok" and refit["n_points"] == 7
    assert refit["bulk_modulus"] == pytest.approx(eos["bulk_modulus"], rel=0.05)
    assert eos["bulk_modulus_derivative_plausible"] and refit["bulk_modulus_derivative_plausible"]
    assert model["warnings"] == []


def test_matcalc_emt_elastic_strain_windows(tmp_path):
    pytest.importorskip("matcalc")
    config = PropertyConfig(("elasticity",), fmax=0.005, elastic_shear_strains=(-0.02, -0.01, 0.01, 0.02),
                            elastic_strain_check=True)
    try:
        payload = run_properties(_cu(), [Member("m0", EMT)], config, tmp_path / "o", engine="emt")
    except DependencyError:
        pytest.skip("matcalc import failed")
    model = payload["models"][0]
    assert model["status"] == "ok", model.get("error")
    elastic = model["results"]["elasticity"]
    assert elastic["strains"]["shear_strains"] == [-0.02, -0.01, 0.01, 0.02]
    check = elastic["strain_check"]
    assert check["reference_strains"]["shear_strains"] == [-0.06, -0.03, 0.03, 0.06]
    tensor = np.asarray(elastic["elastic_tensor"])
    reference = np.asarray(check["reference_elastic_tensor"])
    assert tensor[3, 3] > 0 and reference[3, 3] > 0
    assert check["flagged"] is (check["max_diagonal_relative_change"] > 0.10)


def _matcalc_elasticity(atoms, ion_relaxation, tmp_path, **options):
    config = PropertyConfig(("elasticity",), fmax=1e-4, max_steps=2000, elastic_ion_relaxation=ion_relaxation,
                            **options)
    try:
        payload = run_properties(atoms, [Member("m0", EMT)], config, tmp_path / ion_relaxation, engine="emt")
    except DependencyError:
        pytest.skip("matcalc import failed")
    model = payload["models"][0]
    assert model["status"] == "ok", model.get("error")
    return model["results"]["elasticity"]


def test_matcalc_emt_relaxed_ion_differs_from_clamped_for_hcp(tmp_path):
    pytest.importorskip("matcalc")
    clamped = _matcalc_elasticity(_hcp_cu(), "clamped", tmp_path)
    relaxed = _matcalc_elasticity(_hcp_cu(), "auto", tmp_path, elastic_fmax=1e-4)
    assert clamped["ion_relaxation"] == "clamped" and clamped["deformed_relaxations"] is None
    assert relaxed["ion_relaxation"] == "relaxed" and relaxed["ion_relaxation_selection"]["requested"] == "auto"
    deformed = relaxed["deformed_relaxations"]
    assert deformed["convergence_checked"] and deformed["converged"] and deformed["n_structures"] == 24
    assert deformed["max_force"] <= 1e-4 and deformed["relax_cell"] is False
    c_clamped, c_relaxed = np.asarray(clamped["elastic_tensor"]), np.asarray(relaxed["elastic_tensor"])
    # EMT hcp Cu, like beta-Si3N4: internal relaxation softens C66 (69 -> 52 GPa) and raises C12 (95 -> 112 GPa).
    assert c_relaxed[5, 5] < 0.85 * c_clamped[5, 5]
    assert c_relaxed[0, 1] > 1.1 * c_clamped[0, 1]
    assert relaxed["shear_modulus_vrh"] < clamped["shear_modulus_vrh"]
    # By -6m2 site symmetry, axial strain and the xz/yz shears do not move the atoms.
    assert c_relaxed[2, 2] == pytest.approx(c_clamped[2, 2], rel=0.01)
    assert c_relaxed[3, 3] == pytest.approx(c_clamped[3, 3], rel=0.02)


def test_matcalc_emt_relaxed_ion_equals_clamped_for_fcc(tmp_path):
    pytest.importorskip("matcalc")
    clamped = _matcalc_elasticity(_cu(), "clamped", tmp_path)
    relaxed = _matcalc_elasticity(_cu(), "relaxed", tmp_path)
    assert relaxed["deformed_relaxations"]["converged"]
    assert np.allclose(relaxed["elastic_tensor"], clamped["elastic_tensor"], atol=0.5)


def test_matcalc_deformed_relaxation_hook_is_restored(tmp_path):
    pytest.importorskip("matcalc")
    from matcalc import RelaxCalc, _elasticity

    eq = matcalc_adapter.relax(_hcp_cu(), EMT(), fmax=1e-3, max_steps=500, optimizer="FIRE",
                               member_dir=tmp_path)["atoms"]
    strains = (-0.01, 0.01)
    result = matcalc_adapter.elasticity(eq, EMT(), fmax=0.005, norm_strains=strains, shear_strains=strains,
                                        ion_relaxation="relaxed", max_steps=1)
    assert _elasticity.RelaxCalc is RelaxCalc
    deformed = result["deformed_relaxations"]
    assert deformed["n_structures"] == 12 and deformed["max_steps"] == 1
    assert deformed["converged"] is False and deformed["n_converged"] < 12
    with pytest.raises(ValueError, match="ion_relaxation"):
        matcalc_adapter.elasticity(eq, EMT(), fmax=0.005, ion_relaxation="auto")
    assert _elasticity.RelaxCalc is RelaxCalc
