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
    summarize,
    to_jsonable,
)
from interfaceforge.state import sha256_file


class NoStressCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def _fake_backend(bulk_modulus: float = 140.0, *, tensor_shape: tuple[int, int] = (6, 6), converged: bool = True):
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

    def eos(atoms, calculator, **_):
        scale = getattr(calculator, "scale", 1.0)
        return {"equilibrium_energy": -3.0, "equilibrium_energy_per_atom": -3.0, "equilibrium_volume": 11.6,
                "equilibrium_volume_per_atom": 11.6, "bulk_modulus": bulk_modulus * scale,
                "bulk_modulus_derivative": 4.5, "r2": 0.9999, "volumes": np.linspace(10, 13, 5),
                "units": {"bulk_modulus": "GPa"}}

    def elasticity(atoms, calculator, **_):
        return {"elastic_tensor": np.eye(*tensor_shape) * getattr(calculator, "scale", 1.0),
                "tensor_convention": "voigt", "bulk_modulus_vrh": 140.0, "shear_modulus_vrh": 60.0,
                "youngs_modulus": 157.0, "units": {}}

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
    elastic = models[0]["results"]["elasticity"]
    assert np.asarray(elastic["elastic_tensor"]).shape == (6, 6)
    assert models[0]["results"]["phonon"]["dynamically_stable"]
    out = tmp_path / "o"
    assert json.loads((out / RESULT_FILE).read_text())["inputs_unchanged"] is True
    for name in ("relax.traj", "relaxed.vasp", "phonon.yaml"):
        assert (out / "members" / "m0" / name).is_file()
    assert list(cwd.iterdir()) == []  # MatCalc's default phonon.yaml-in-CWD is redirected
