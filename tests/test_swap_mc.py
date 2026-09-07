from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from interfaceforge.errors import SafetyError
from interfaceforge.swap_mc import (
    CAVEATS,
    CITATION,
    MlipRelaxer,
    RelaxOutcome,
    RelaxTier,
    export_candidates,
    initial_occupation,
    propose_swap,
    random_configurations,
    resolve_swap_sites,
    run_search,
    run_searches,
    select_candidates,
    write_archive,
)

# stack: (element, z) along c. Lower slab is Si/N, upper slab is Ti/N.
_STACK = [
    ("Si", 0.0), ("N", 1.5), ("Si", 3.0), ("N", 4.5), ("Si", 6.0), ("N", 7.5),
    ("Ti", 9.5), ("N", 11.0), ("Ti", 12.5), ("N", 14.0), ("Ti", 15.5), ("N", 17.0),
]
_A = 3.0
_NX = _NY = 3


def _interface(path: Path, *, n_oxygen: int = 0, freeze_bottom: bool = True) -> Atoms:
    positions: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    for sym, z in _STACK:
        for i in range(_NX):
            for j in range(_NY):
                positions.append((i * _A, j * _A, z))
                symbols.append(sym)
    atoms = Atoms(symbols=symbols, positions=positions, cell=[_NX * _A, _NY * _A, 20.0], pbc=True)
    if freeze_bottom:
        atoms.set_constraint(FixAtoms(mask=atoms.positions[:, 2] < 2.0))
    if n_oxygen:
        movable_n = [
            idx
            for idx, sym in enumerate(atoms.get_chemical_symbols())
            if sym == "N" and atoms.positions[idx, 2] > 2.0
        ]
        # spread the initial O across high layers so a search has room to improve
        chosen = movable_n[-1 : -1 - n_oxygen * 4 : -4][:n_oxygen]
        syms = atoms.get_chemical_symbols()
        for idx in chosen:
            syms[idx] = "O"
        atoms.set_chemical_symbols(syms)
    if path is not None:
        write(str(path), atoms, format="vasp", direct=True, vasp5=True)
    return atoms


class _FakeRelaxer:
    """Analytic lattice-gas energy: O prefers low anion layers and clustering.

    No geometry change -- exercises the search / archive / selection logic
    without mace-torch or deepmd-kit.
    """

    def __init__(self, eligible, layer_of, substituent="O", *, field=1.0, coupling=0.5):
        self.eligible = list(eligible)
        self.layer_of = dict(layer_of)
        self.substituent = substituent
        self.field = field
        self.coupling = coupling
        self.calls = 0

    def _energy(self, atoms) -> float:
        syms = atoms.get_chemical_symbols()
        o_sites = [i for i in self.eligible if syms[i] == self.substituent]
        e = self.field * sum(self.layer_of[i] for i in o_sites)
        same = sum(
            1
            for a in range(len(o_sites))
            for b in range(a + 1, len(o_sites))
            if self.layer_of[o_sites[a]] == self.layer_of[o_sites[b]]
        )
        return e - self.coupling * same

    def relax(self, atoms, tier: RelaxTier) -> RelaxOutcome:
        self.calls += 1
        return RelaxOutcome(
            atoms=atoms.copy(),
            energy=self._energy(atoms),
            converged=tier.max_steps > 0,
            steps=min(tier.max_steps, 4),
            max_force=0.02,
        )

    def uncertainty(self, atoms) -> float:
        return 0.05

    def identity(self) -> dict:
        return {"family": "fake", "models": []}


def _sites(atoms, **kw):
    return resolve_swap_sites(atoms, **kw)


# --------------------------------------------------------------------- site masks


def test_resolve_swap_sites_excludes_cations_and_frozen(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=3)
    sites = _sites(atoms)
    symbols = np.asarray(atoms.get_chemical_symbols())
    assert all(symbols[i] in ("N", "O") for i in sites.eligible)
    # bottom Si + first N layer (z < 2.0) are frozen and must be excluded
    assert sites.excluded_frozen
    assert all(atoms.positions[i, 2] >= 2.0 for i in sites.eligible)
    # oxygen already present is the fixed composition
    assert sites.n_substituent == 3
    assert set(sites.initial_substituent).issubset(set(sites.eligible))


def test_resolve_swap_sites_layer_groups_and_interface_plane(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=2)
    sites = _sites(atoms, move_classes="layer")
    # five eligible anion planes (z = 4.5, 7.5, 11, 14, 17)
    assert len(sites.groups) == 5
    assert sum(len(v) for v in sites.groups.values()) == len(sites.eligible)
    # contact plane sits between the topmost Si plane (6.0) and the lowest Ti plane (9.5)
    assert sites.interface_z == pytest.approx((6.0 + 9.5) / 2.0)


def test_region_filter_splits_lower_and_upper(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=0, freeze_bottom=False)
    lower = _sites(atoms, region="lower", substituent="O", host="N")
    upper = _sites(atoms, region="upper", substituent="O", host="N")
    assert set(lower.eligible).isdisjoint(upper.eligible)
    assert all(atoms.positions[i, 2] < 9.0 for i in lower.eligible)
    assert all(atoms.positions[i, 2] > 8.0 for i in upper.eligible)


def test_interface_band_narrows_the_window(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", freeze_bottom=False)
    band = _sites(atoms, interface_band=4.0)
    assert 0 < len(band.eligible) < len(_sites(atoms).eligible)
    assert all(abs(atoms.positions[i, 2] - band.interface_z) <= 2.0 + 1e-9 for i in band.eligible)


def test_resolve_rejects_too_few_sites(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp")
    with pytest.raises(SafetyError):
        _sites(atoms, z_window=(0.0, 0.1))


# ---------------------------------------------------------------- occupation moves


def test_initial_occupation_modes(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=4)
    sites = _sites(atoms)
    keep = initial_occupation(sites, mode="keep")
    assert keep == frozenset(sites.initial_substituent)
    scrambled = initial_occupation(sites, mode="randomize", seed=1)
    assert len(scrambled) == 4 and scrambled.issubset(set(sites.eligible))

    clean = _sites(_interface(tmp_path / "clean.vasp", n_oxygen=0))
    introduced = initial_occupation(clean, mode="introduce", introduce=5, seed=2)
    assert len(introduced) == 5
    with pytest.raises(SafetyError):
        initial_occupation(sites, mode="introduce", introduce=2)  # O already present


def test_propose_swap_conserves_composition(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=6)
    sites = _sites(atoms)
    rng = np.random.default_rng(0)
    occ = initial_occupation(sites, mode="keep")
    for _ in range(200):
        s, h, mode = propose_swap(rng, occ, sites, p_interclass=0.5)
        assert s in occ and h not in occ
        assert mode in ("intra-class", "inter-class", "fallback")
        occ = frozenset((occ - {s}) | {h})
        assert len(occ) == 6


def test_random_configurations_are_distinct_and_fixed_composition(tmp_path):
    sites = _sites(_interface(tmp_path / "iface.vasp", n_oxygen=5))
    configs = random_configurations(sites, count=8, seed=3)
    assert len(configs) == 8
    assert len({tuple(sorted(c)) for c in configs}) == 8
    assert all(len(c) == 5 for c in configs)


# ---------------------------------------------------------------------- MC search


def test_run_search_finds_lower_energy_and_records_candidates(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=5)
    sites = _sites(atoms)
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    candidates: dict = {}
    result = run_search(
        atoms,
        sites,
        relaxer,
        steps=150,
        kt_ev=0.05,
        seed=7,
        screen=RelaxTier(0.1, 20, "screen"),
        refine=RelaxTier(0.01, 200, "refine"),
        refine_accepted=True,
        candidates=candidates,
    )
    assert result.best_energy <= result.initial_energy
    assert len(result.trajectory) == 150
    assert 0.0 < result.acceptance_rate <= 1.0
    assert result.n_evaluated == 151
    # composition is conserved for every distinct arrangement seen
    assert {len(key) for key in candidates} == {5}
    # the global best was refined with the strict tier
    best = candidates[result.best_occupation]
    assert best.energy_refine is not None
    assert "best" in best.roles and "refined" in best.roles
    assert CAVEATS[0] in result.warnings


def test_run_searches_aggregates_seeds_and_baseline(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=4)
    sites = _sites(atoms)
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    aggregate = run_searches(
        atoms,
        sites,
        relaxer,
        seeds=[1, 2],
        steps=80,
        kt_ev=0.04,
        screen=RelaxTier(0.1, 15, "screen"),
        refine=RelaxTier(0.01, 150, "refine"),
        random_baseline=3,
        baseline_seed=99,
    )
    assert len(aggregate["per_seed"]) == 2
    assert aggregate["consensus"]["n_seeds"] == 2
    assert set(aggregate["seed_best_energies_ev"]) == {"1", "2"}
    roles = {r for rec in aggregate["candidates"].values() for r in rec.roles}
    assert "random-baseline" in roles
    # consensus is the archive minimum
    assert aggregate["consensus"]["energy_ev"] == pytest.approx(
        min(rec.energy for rec in aggregate["candidates"].values())
    )


def test_run_search_rejects_bad_kt(tmp_path):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=3)
    sites = _sites(atoms)
    with pytest.raises(SafetyError):
        run_search(
            atoms, sites, _FakeRelaxer(sites.eligible, sites.layer_of),
            steps=5, kt_ev=0.0, seed=1,
            screen=RelaxTier(0.1, 10, "s"), refine=RelaxTier(0.01, 50, "r"),
        )


# ------------------------------------------------------------------------ archive


def _run(tmp_path, **kw):
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=kw.pop("n_oxygen", 5))
    sites = _sites(atoms)
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    aggregate = run_searches(
        atoms, sites, relaxer,
        seeds=kw.pop("seeds", [11, 23]),
        steps=kw.pop("steps", 120),
        kt_ev=0.04,
        screen=RelaxTier(0.1, 15, "screen"),
        refine=RelaxTier(0.01, 150, "refine"),
        random_baseline=kw.pop("random_baseline", 4),
        baseline_seed=5,
    )
    run_dir = tmp_path / "run"
    result = write_archive(
        run_dir, aggregate,
        config={"steps": 120, "kt_ev": 0.04},
        sites=sites,
        relaxer_identity=relaxer.identity(),
        input_structure=tmp_path / "iface.vasp",
        **kw,
    )
    return run_dir, result, aggregate


def test_write_archive_layout_and_citation(tmp_path):
    run_dir, result, aggregate = _run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["quantity"] == "chemical_ordering_search"
    assert manifest["citation"]["software"]["name"] == "PAIPAI"
    assert manifest["citation"]["paper"]["pages"] == "114752"
    assert manifest["caveats"] == list(CAVEATS)
    assert manifest["sites"]["n_eligible"] == manifest["sites"]["n_eligible"]

    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()]
    assert rows and rows == sorted(rows, key=lambda r: r["energy_ev"])
    assert all(r["n_substituent"] == 5 for r in rows)
    assert (run_dir / "trajectory_seed_11.csv").is_file()
    assert (run_dir / "trajectory_seed_23.csv").is_file()
    assert (run_dir / "summary.md").read_text().count("PAIPAI") >= 1

    geoms = list((run_dir / "candidates").glob("*.extxyz"))
    assert geoms
    assert result["geometries_written"] == len(geoms)


def test_write_archive_tolerates_a_prior_dry_run_plan(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "plan.json").write_text("{}", encoding="utf-8")
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=3)
    sites = _sites(atoms)
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    aggregate = run_searches(
        atoms, sites, relaxer, seeds=[1], steps=10, kt_ev=0.05,
        screen=RelaxTier(0.1, 10, "s"), refine=RelaxTier(0.01, 50, "r"),
    )
    result = write_archive(
        run_dir, aggregate, config={}, sites=sites,
        relaxer_identity=relaxer.identity(), input_structure=tmp_path / "iface.vasp",
    )
    assert Path(result["manifest"]).is_file()


def test_write_archive_guards_foreign_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "important.txt").write_text("keep me", encoding="utf-8")
    atoms = _interface(tmp_path / "iface.vasp", n_oxygen=3)
    sites = _sites(atoms)
    relaxer = _FakeRelaxer(sites.eligible, sites.layer_of)
    aggregate = run_searches(
        atoms, sites, relaxer, seeds=[1], steps=10, kt_ev=0.05,
        screen=RelaxTier(0.1, 10, "s"), refine=RelaxTier(0.01, 50, "r"),
    )
    with pytest.raises(SafetyError):
        write_archive(
            run_dir, aggregate, config={}, sites=sites,
            relaxer_identity=relaxer.identity(), input_structure=tmp_path / "iface.vasp",
        )


# ---------------------------------------------------------------------- shortlist


def test_select_candidates_shortlist_roles_and_outputs(tmp_path):
    run_dir, _, _ = _run(tmp_path)
    payload = select_candidates(run_dir, count=8, low_fraction=0.5)
    assert payload["n_selected"] >= 1
    roles = {row["role"] for row in payload["shortlist"]}
    assert "low-energy" in roles
    assert {"baseline", "diverse"} & roles
    assert Path(payload["outputs"]["csv"]).is_file()
    assert Path(payload["outputs"]["json"]).is_file()
    # sorted by energy, deltas relative to the archive minimum
    energies = [row["energy_ev"] for row in payload["shortlist"]]
    assert energies == sorted(energies)
    assert payload["shortlist"][0]["delta_vs_best_ev"] == pytest.approx(0.0, abs=1e-9)


def test_select_candidates_validates_inputs(tmp_path):
    run_dir, _, _ = _run(tmp_path)
    with pytest.raises(SafetyError):
        select_candidates(run_dir, count=0)
    with pytest.raises(SafetyError):
        select_candidates(tmp_path / "nonexistent")


# ------------------------------------------------------------------------- export


def test_export_candidates_writes_poscar_dirs(tmp_path):
    run_dir, _, _ = _run(tmp_path)
    select_candidates(run_dir, count=6, low_fraction=0.5)
    export_dir = tmp_path / "export"
    result = export_candidates(run_dir, export_dir)
    assert result["n_exported"] >= 1
    manifest = json.loads((export_dir / "manifest.json").read_text())
    assert manifest["quantity"] == "chemical_ordering_export"
    assert manifest["citation"]["software"]["url"] == CITATION["software"]["url"]
    for entry in result["candidates"]:
        target = Path(entry["directory"])
        assert (target / "POSCAR").is_file()
        ordering = json.loads((target / "ordering.json").read_text())
        assert ordering["n_substituent"] == 5
        atoms = read(str(target / "POSCAR"))
        assert sum(1 for s in atoms.get_chemical_symbols() if s == "O") == 5
        # the interface's frozen substrate layers survive archive + export
        frozen = {int(i) for c in atoms.constraints for i in c.get_indices()}
        assert frozen and all(atoms.positions[i, 2] < 2.0 for i in frozen)


def test_export_candidates_by_id_and_force_guard(tmp_path):
    run_dir, _, _ = _run(tmp_path)
    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()]
    with_geom = [r["cand_id"] for r in rows if "geometry" in r][:2]
    export_dir = tmp_path / "export"
    export_candidates(run_dir, export_dir, cand_ids=with_geom)
    assert {p.name for p in export_dir.iterdir() if p.is_dir()} == set(with_geom)
    with pytest.raises(SafetyError):
        export_candidates(run_dir, export_dir, cand_ids=with_geom)
    export_candidates(run_dir, export_dir, cand_ids=with_geom, force=True)


# ---------------------------------------------------------------------------- CLI


def test_cli_sites_and_dry_run(tmp_path, capsys):
    from interfaceforge import cli

    structure = tmp_path / "iface.vasp"
    _interface(structure, n_oxygen=4)

    assert cli.main(["swap-mc", "sites", str(structure), "--move-classes", "layer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_eligible"] > 0 and payload["initial_substituent_count"] == 4

    out = tmp_path / "run"
    assert cli.main(["swap-mc", "run", str(structure), str(out), "--dry-run", "--steps", "50"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == "dry-run"
    assert plan["config"]["steps"] == 50
    assert plan["citation"]["software"]["name"] == "PAIPAI"


def test_cli_run_without_backend_errors(tmp_path):
    from interfaceforge import cli

    structure = tmp_path / "iface.vasp"
    _interface(structure, n_oxygen=3)
    assert cli.main(["swap-mc", "run", str(structure), str(tmp_path / "run")]) == 2


def test_mlip_relaxer_requires_models():
    with pytest.raises(SafetyError):
        MlipRelaxer("mace", [])
