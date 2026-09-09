from __future__ import annotations

import json
from pathlib import Path

import pytest

from interfaceforge.errors import SafetyError
from interfaceforge.ordering_dft import (
    _read_poscar,
    collect_ordering_dft,
    compare_ordering,
    launch_ordering_dft,
    prepare_ordering_dft,
)

_LATTICE = ("   10.0000000000    0.0000000000    0.0000000000\n"
            "    0.0000000000   10.0000000000    0.0000000000\n"
            "    0.0000000000    0.0000000000   24.0000000000\n")

_REFERENCE_INCAR = """SYSTEM = reference AIMD
# ---------------- electronic ----------------
ENCUT  = 520
PREC   = Accurate
ISPIN  = 2
LDAU   = .TRUE.
NCORE  = 8
# ---------------- dynamics ----------------
IBRION = 0
NSW    = 5000
POTIM  = 2.0
TEBEG  = 1200
SMASS  = 0
MDALGO = 2
ML_LMLFF = .TRUE.
ML_MODE  = train
"""


def _poscar(species: list[str], counts: list[int], *, frozen_element: str = "Si") -> str:
    """A slab POSCAR whose frozen atoms are one whole species, whatever the block order."""

    symbols = [symbol for symbol, count in zip(species, counts, strict=True) for _ in range(count)]
    rows = []
    for index, symbol in enumerate(symbols):
        z = 0.02 * index
        flag = "F F F" if symbol == frozen_element else "T T T"
        rows.append(f"  0.1000000000  0.2000000000  {z:.10f}   {flag}")
    return (
        "ordering candidate\n"
        "   1.00000000000000\n"
        + _LATTICE
        + "  " + "  ".join(species) + "\n"
        + "  " + "  ".join(str(value) for value in counts) + "\n"
        "Selective dynamics\n"
        "Direct\n"
        + "\n".join(rows)
        + "\n"
    )


def _potcar(elements: list[str]) -> str:
    return "".join(
        f"  PAW_PBE {element} 08Apr2002\n   VRHFIN ={element}: s2p3\n End of Dataset\n"
        for element in elements
    )


def _reference(root: Path) -> Path:
    source = root / "reference"
    source.mkdir()
    (source / "INCAR").write_text(_REFERENCE_INCAR, encoding="utf-8")
    (source / "KPOINTS").write_text("auto\n0\nGamma\n3 3 1\n0 0 0\n", encoding="utf-8")
    (source / "POTCAR").write_text(_potcar(["N", "O", "Si", "Ti"]), encoding="utf-8")
    (source / "runvasp.sh").write_text("#!/bin/bash\nsrun vasp_std\n", encoding="utf-8")
    return source


def _export(root: Path, *, candidates: dict[str, dict] | None = None) -> Path:
    """A stand-in for an 'iface swap-mc export' tree: differing species order on purpose."""

    export = root / "export"
    export.mkdir()
    spec = candidates or {
        "cand_a": {"roles": ["initial"], "role": "baseline", "energy_ev": -100.00,
                   "species": ["Si", "Ti", "N", "O"], "counts": [4, 4, 6, 2]},
        "cand_b": {"roles": ["random-baseline"], "role": "baseline", "energy_ev": -100.10,
                   "species": ["Si", "Ti", "O", "N"], "counts": [4, 4, 2, 6]},
        "cand_c": {"roles": ["random-baseline"], "role": "baseline", "energy_ev": -100.20,
                   "species": ["N", "O", "Si", "Ti"], "counts": [6, 2, 4, 4]},
        "cand_d": {"roles": ["accepted"], "role": "low-energy", "energy_ev": -100.60,
                   "species": ["Si", "Ti", "N", "O"], "counts": [4, 4, 6, 2]},
        "cand_e": {"roles": ["accepted"], "role": "low-energy", "energy_ev": -100.50,
                   "species": ["Si", "Ti", "N", "O"], "counts": [4, 4, 6, 2]},
    }
    entries = []
    for cand_id, item in spec.items():
        directory = export / cand_id
        directory.mkdir()
        (directory / "POSCAR").write_text(
            item.get("poscar") or _poscar(item["species"], item["counts"]), encoding="utf-8"
        )
        (directory / "ordering.json").write_text(
            json.dumps({
                "cand_id": cand_id,
                "role": item["role"],
                "roles": item["roles"],
                "n_substituent": 2,
                "substituent_sites": [1, 2],
                "energy_ev": item["energy_ev"],
                "energy_refine_ev": item["energy_ev"],
                "force_std_ev_ang": 0.05,
            }),
            encoding="utf-8",
        )
        entries.append({"cand_id": cand_id, "directory": str(directory)})
    (export / "manifest.json").write_text(
        json.dumps({"candidates": entries}), encoding="utf-8"
    )
    return export


def _outcar(energy: float, *, converged: bool = True) -> str:
    body = (
        " FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
        f"  free  energy   TOTEN  =      {energy:.6f} eV\n"
        f"  energy  without entropy=     {energy + 0.01:.6f}  "
        f"energy(sigma->0) =     {energy:.6f}\n"
    )
    if converged:
        body += " reached required accuracy - stopping structural energy minimisation\n"
    return body + " General timing and accounting informations for this job\n"


def _prepare(tmp_path: Path, **kwargs):
    source = _reference(tmp_path)
    export = _export(tmp_path)
    return prepare_ordering_dft(
        export, tmp_path / "dft", reference=source, stages=("static", "relax"), **kwargs
    )


class TestPrepare:
    def test_every_candidate_shares_identical_inputs(self, tmp_path: Path) -> None:
        result = _prepare(tmp_path)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        assert result["n_candidates"] == 5
        assert result["runs"] == 10

        for stage in ("static", "relax"):
            incars = {
                (tmp_path / "dft" / stage / cand / "INCAR").read_text(encoding="utf-8")
                for cand in ("cand_a", "cand_b", "cand_c", "cand_d", "cand_e")
            }
            # Only the SYSTEM line may differ between candidates.
            bodies = {
                "\n".join(line for line in text.splitlines() if not line.startswith("SYSTEM"))
                for text in incars
            }
            assert len(bodies) == 1
            kpoints = {
                (tmp_path / "dft" / stage / cand / "KPOINTS").read_bytes()
                for cand in ("cand_a", "cand_d")
            }
            assert len(kpoints) == 1
        assert manifest["shared_inputs"]["species_order"] == ["N", "O", "Si", "Ti"]
        assert manifest["shared_inputs"]["composition"] == {"Si": 4, "Ti": 4, "N": 6, "O": 2}
        assert len(manifest["shared_inputs"]["incar_body_sha256"]) == 2

    def test_incar_stages_and_inherited_settings(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        static = (tmp_path / "dft" / "static" / "cand_a" / "INCAR").read_text(encoding="utf-8")
        relax = (tmp_path / "dft" / "relax" / "cand_a" / "INCAR").read_text(encoding="utf-8")

        assert "ENCUT           = 520" in static or "ENCUT  = 520" in static
        assert "ISPIN" in static and "LDAU" in static and "NCORE" in static
        for text in (static, relax):
            assert "ML_LMLFF" not in text and "ML_MODE" not in text
            assert "TEBEG" not in text and "SMASS" not in text and "MDALGO" not in text
            assert "ISYM            = 0" in text
            assert "EDIFF           = 1E-6" in text
        assert "IBRION          = -1" in static and "NSW             = 0" in static
        assert "IBRION          = 2" in relax and "NSW             = 99" in relax
        assert "EDIFFG          = -0.02" in relax
        assert "EDIFFG" not in static
        # the reference AIMD timestep must not leak into a single point
        assert "POTIM" not in static
        assert "POTIM           = 0.20" in relax

    def test_poscars_are_canonicalized_and_keep_constraints(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        for cand in ("cand_a", "cand_b", "cand_c"):
            parsed = _read_poscar(tmp_path / "dft" / "static" / cand / "POSCAR")
            assert parsed.species == ("N", "O", "Si", "Ti")
            assert parsed.counts == (6, 2, 4, 4)
            assert parsed.selective is True
            frozen = [
                index for index, flag in enumerate(parsed.flags)
                if flag and all(value.upper() == "F" for value in flag)
            ]
            # N6 O2 Si4 Ti4: the frozen atoms must still be exactly the Si block.
            assert frozen == [8, 9, 10, 11]

    def test_potcar_is_shared_and_matches_the_species_order(self, tmp_path: Path) -> None:
        result = _prepare(tmp_path)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        digests = {
            (tmp_path / "dft" / "static" / cand / "POTCAR").read_bytes()
            for cand in ("cand_a", "cand_b")
        }
        assert len(digests) == 1
        assert manifest["shared_inputs"]["potcar"]["origin"] == "reference"
        assert (tmp_path / "dft" / "static" / "cand_a" / "runvasp.sh").is_file()

    def test_rejects_a_different_cell(self, tmp_path: Path) -> None:
        source = _reference(tmp_path)
        export = _export(tmp_path)
        bad = (export / "cand_b" / "POSCAR").read_text(encoding="utf-8")
        (export / "cand_b" / "POSCAR").write_text(bad.replace("24.0000000000", "25.0000000000"), encoding="utf-8")
        with pytest.raises(SafetyError, match="cell differs"):
            prepare_ordering_dft(export, tmp_path / "dft", reference=source)

    def test_rejects_a_different_composition(self, tmp_path: Path) -> None:
        source = _reference(tmp_path)
        export = _export(tmp_path)
        (export / "cand_b" / "POSCAR").write_text(
            _poscar(["Si", "Ti", "N", "O"], [4, 4, 5, 3]), encoding="utf-8"
        )
        with pytest.raises(SafetyError, match="composition"):
            prepare_ordering_dft(export, tmp_path / "dft", reference=source)

    def test_rejects_a_different_frozen_layer_definition(self, tmp_path: Path) -> None:
        source = _reference(tmp_path)
        export = _export(tmp_path)
        (export / "cand_b" / "POSCAR").write_text(
            _poscar(["Si", "Ti", "N", "O"], [4, 4, 6, 2], frozen_element="Ti"), encoding="utf-8"
        )
        with pytest.raises(SafetyError, match="frozen atoms"):
            prepare_ordering_dft(export, tmp_path / "dft", reference=source)

    def test_refuses_to_overwrite_a_prepared_tree(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        source = tmp_path / "reference"
        export = tmp_path / "export"
        with pytest.raises(SafetyError, match="refusing to overwrite"):
            prepare_ordering_dft(export, tmp_path / "dft", reference=source)

    def test_warns_when_the_reference_incar_has_no_cutoff(self, tmp_path: Path) -> None:
        source = _reference(tmp_path)
        (source / "INCAR").write_text(
            _REFERENCE_INCAR.replace("ENCUT  = 520\n", ""), encoding="utf-8"
        )
        result = prepare_ordering_dft(_export(tmp_path), tmp_path / "dft", reference=source)
        assert any("ENCUT" in warning for warning in result["warnings"])


class TestLaunch:
    def test_dry_run_lists_every_run_and_submits_nothing(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        result = launch_ordering_dft(tmp_path / "dft", stage="static")
        assert result["mode"] == "dry-run"
        assert result["runs"] == 5
        assert result["preflight"] == "PASS"
        assert not (tmp_path / "dft" / "ordering_dft_launch_static.json").exists()

    def test_edited_inputs_block_the_launch(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        incar = tmp_path / "dft" / "static" / "cand_b" / "INCAR"
        incar.write_text(incar.read_text(encoding="utf-8") + "ENCUT = 400\n", encoding="utf-8")
        with pytest.raises(SafetyError, match="changed since dft-prepare"):
            launch_ordering_dft(tmp_path / "dft", stage="static")

    def test_existing_output_is_skipped_not_resubmitted(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        (tmp_path / "dft" / "static" / "cand_a" / "OUTCAR").write_text(_outcar(-100.0), encoding="utf-8")
        result = launch_ordering_dft(tmp_path / "dft", stage="static")
        assert result["runs"] == 4
        assert [row["cand_id"] for row in result["skipped"]] == ["cand_a"]

    def test_limit_bounds_the_batch(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        result = launch_ordering_dft(tmp_path / "dft", stage="static", limit=2)
        assert result["runs"] == 2

    def test_submits_each_run_once_with_execute(self, tmp_path: Path, monkeypatch) -> None:
        _prepare(tmp_path)
        submitted: list[str] = []

        def fake_submit(folder, launcher, **kwargs):
            submitted.append(Path(folder).name)
            return f"{1000 + len(submitted)}"

        monkeypatch.setattr("interfaceforge.ordering_dft.submit_run", fake_submit)
        result = launch_ordering_dft(tmp_path / "dft", stage="static", execute=True)
        assert result["submitted"] == 5
        assert sorted(submitted) == ["cand_a", "cand_b", "cand_c", "cand_d", "cand_e"]
        payload = json.loads((tmp_path / "dft" / "ordering_dft_launch_static.json").read_text(encoding="utf-8"))
        assert payload["status"] == "SUBMITTED"

    def test_unprepared_stage_is_refused(self, tmp_path: Path) -> None:
        source = _reference(tmp_path)
        prepare_ordering_dft(_export(tmp_path), tmp_path / "dft", reference=source, stages=("static",))
        with pytest.raises(SafetyError, match="no prepared 'relax' stage"):
            launch_ordering_dft(tmp_path / "dft", stage="relax")


def _finish(tmp_path: Path, energies: dict[str, float], *, stage: str = "static") -> None:
    for cand_id, energy in energies.items():
        (tmp_path / "dft" / stage / cand_id / "OUTCAR").write_text(_outcar(energy), encoding="utf-8")


class TestCollectAndCompare:
    energies = {
        "cand_a": -500.00,   # initial -> the reference configuration
        "cand_b": -500.08,   # random
        "cand_c": -500.12,   # random
        "cand_d": -500.55,   # searched
        "cand_e": -500.45,   # searched
    }

    def test_collect_reports_usable_and_unfinished_runs(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, {key: value for key, value in self.energies.items() if key != "cand_e"})
        payload = collect_ordering_dft(tmp_path / "dft", stage="static")
        assert payload["n_runs"] == 5
        assert payload["n_usable"] == 4
        rows = {row["cand_id"]: row for row in payload["runs"]}
        assert rows["cand_d"]["dft_energy_ev"] == pytest.approx(-500.55)
        assert rows["cand_e"]["usable"] is False
        assert Path(payload["outputs"]["csv"]).is_file()

    def test_compare_uses_the_initial_arrangement_as_reference(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, self.energies)
        payload = compare_ordering(tmp_path / "dft", stage="static", figure=False)
        stats = payload["statistics"]
        assert stats["reference_cand_id"] == "cand_a"
        assert stats["reference_policy"] == "initial-arrangement"
        assert stats["n_compared"] == 5
        rows = {row["cand_id"]: row for row in payload["candidates"]}
        assert rows["cand_a"]["delta_dft"] == pytest.approx(0.0)
        assert rows["cand_d"]["delta_dft"] == pytest.approx(-0.55)
        assert rows["cand_d"]["delta_mlip"] == pytest.approx(-0.60)
        assert rows["cand_d"]["residual"] == pytest.approx(-0.05)

    def test_compare_reports_the_searched_versus_random_gain(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, self.energies)
        payload = compare_ordering(tmp_path / "dft", stage="static", figure=False)
        verdict = payload["searched_vs_random"]
        assert verdict["testable"] is True
        assert verdict["n_searched"] == 2 and verdict["n_random"] == 2
        # best searched (-0.55) minus mean random ((-0.08 + -0.12)/2 = -0.10)
        assert verdict["dft_gain"] == pytest.approx(-0.45)
        assert verdict["mlip_gain"] == pytest.approx(-0.45)
        assert verdict["sign_agrees"] is True
        assert verdict["dft_confirms_stabilization"] is True

    def test_compare_flags_a_search_that_exploits_model_error(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        # DFT says the MLIP's favourite arrangement is the worst one.
        _finish(tmp_path, {**self.energies, "cand_d": -499.50, "cand_e": -499.60})
        payload = compare_ordering(tmp_path / "dft", stage="static", figure=False)
        assert payload["searched_vs_random"]["sign_agrees"] is False
        assert payload["statistics"]["mlip_best_is_dft_best"] is False
        assert any("exploiting model error" in line for line in payload["interpretation"])

    def test_compare_statistics_and_outputs(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, self.energies)
        payload = compare_ordering(tmp_path / "dft", stage="static", figure=False)
        stats = payload["statistics"]
        assert stats["spearman_rho"] == pytest.approx(1.0)
        assert stats["pearson_r"] > 0.99
        assert stats["pairwise_order_agreement"] == pytest.approx(1.0)
        assert stats["mlip_best_is_dft_best"] is True
        assert stats["mae"] < 0.06
        markdown = Path(payload["outputs"]["markdown"]).read_text(encoding="utf-8")
        assert "Searched vs random" in markdown and "cand_d" in markdown
        assert Path(payload["outputs"]["csv"]).is_file()

    def test_compare_per_atom_rescales(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, self.energies)
        payload = compare_ordering(tmp_path / "dft", stage="static", per_atom=True, figure=False)
        rows = {row["cand_id"]: row for row in payload["candidates"]}
        assert payload["statistics"]["unit"] == "meV/atom"
        assert rows["cand_d"]["delta_dft"] == pytest.approx(-0.55 * 1000 / 16)

    def test_compare_needs_at_least_two_finished_runs(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        _finish(tmp_path, {"cand_a": -500.0})
        with pytest.raises(SafetyError, match="nothing to compare"):
            compare_ordering(tmp_path / "dft", stage="static", figure=False)

    def test_unconverged_relaxation_is_excluded(self, tmp_path: Path) -> None:
        _prepare(tmp_path)
        for cand_id, energy in self.energies.items():
            (tmp_path / "dft" / "relax" / cand_id / "OUTCAR").write_text(
                _outcar(energy, converged=cand_id != "cand_d"), encoding="utf-8"
            )
        payload = collect_ordering_dft(tmp_path / "dft", stage="relax")
        rows = {row["cand_id"]: row for row in payload["runs"]}
        assert rows["cand_d"]["usable"] is False
        assert rows["cand_e"]["usable"] is True


# ------------------------------------------------------- real swap-MC export -> VASP


class _FakeRelaxer:
    """Analytic energy so the search runs without mace-torch or deepmd-kit."""

    def __init__(self, eligible, layer_of):
        self.eligible = list(eligible)
        self.layer_of = dict(layer_of)

    def relax(self, atoms, tier):
        from interfaceforge.swap_mc import RelaxOutcome

        symbols = atoms.get_chemical_symbols()
        energy = sum(self.layer_of[index] for index in self.eligible if symbols[index] == "O")
        return RelaxOutcome(atoms=atoms.copy(), energy=float(energy), converged=True, steps=3, max_force=0.02)

    def uncertainty(self, atoms):
        return 0.05

    def identity(self):
        return {"family": "fake", "models": []}


def test_prepare_consumes_a_real_swap_mc_export(tmp_path: Path) -> None:
    """The export writes interleaved N/O species blocks; prepare must canonicalize them."""

    ase = pytest.importorskip("ase")
    from ase.constraints import FixAtoms

    from interfaceforge.swap_mc import (
        RelaxTier,
        export_candidates,
        resolve_swap_sites,
        run_searches,
        select_candidates,
        write_archive,
    )

    stack = [("Si", 0.0), ("N", 1.5), ("Si", 3.0), ("N", 4.5),
             ("Ti", 6.5), ("N", 8.0), ("Ti", 9.5), ("N", 11.0)]
    positions, symbols = [], []
    for symbol, z in stack:
        for i in range(3):
            for j in range(3):
                positions.append((i * 3.0, j * 3.0, z))
                symbols.append(symbol)
    atoms = ase.Atoms(symbols=symbols, positions=positions, cell=[9.0, 9.0, 20.0], pbc=True)
    atoms.set_constraint(FixAtoms(mask=atoms.positions[:, 2] < 2.0))
    movable_n = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "N" and atoms.positions[i, 2] > 2.0]
    updated = atoms.get_chemical_symbols()
    for index in movable_n[:4]:
        updated[index] = "O"
    atoms.set_chemical_symbols(updated)

    sites = resolve_swap_sites(atoms, substituent="O", host="N", cations=("Si", "Ti"), move_classes="layer")
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    aggregate = run_searches(
        atoms, sites, relaxer, seeds=[3], steps=25, kt_ev=0.05,
        screen=RelaxTier(0.10, 30, "screen"), refine=RelaxTier(0.01, 400, "refine"),
        random_baseline=3, baseline_seed=11,
    )
    run_dir = tmp_path / "run"
    write_archive(run_dir, aggregate, config={"steps": 25}, sites=sites,
                  relaxer_identity=relaxer.identity(), input_structure=tmp_path / "input.vasp", geom_cap=40)
    select_candidates(run_dir, count=6, low_fraction=0.5)
    export = tmp_path / "export"
    export_candidates(run_dir, export)

    result = prepare_ordering_dft(export, tmp_path / "dft", reference=_reference(tmp_path))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["shared_inputs"]["species_order"] == ["N", "O", "Si", "Ti"]
    assert {entry["group"] for entry in manifest["candidates"]} & {"random", "searched"}

    for entry in manifest["candidates"]:
        parsed = _read_poscar(tmp_path / "dft" / "static" / entry["cand_id"] / "POSCAR")
        assert parsed.species == ("N", "O", "Si", "Ti")
        assert parsed.selective is True
        # the frozen substrate survives archive -> export -> canonical rewrite
        assert any(flag and flag[0].upper() == "F" for flag in parsed.flags)
