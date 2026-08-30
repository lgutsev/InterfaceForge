from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.build import fcc110
from ase.io import read, write

from interfaceforge.surface import (
    analyze_surface,
    audit_surface_runs,
    build_surface_campaign,
    optimize_surface_cell,
    plan_surface_campaign,
    select_surface_candidates,
)
from interfaceforge.surface.magnetism import assign_superexchange_afm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "nio_m110_hydroxylation"
REFERENCE_CAMPAIGN = NOTEBOOK / "surface_campaign.yaml"
REFERENCE_SLAB = NOTEBOOK / "inputs" / "NiO_110_AFM_compromise.POSCAR"
REFERENCE_LIGAND = NOTEBOOK / "inputs" / "Me4PACz.xyz"


def _campaign(tmp_path: Path) -> Path:
    payload = yaml.safe_load(REFERENCE_CAMPAIGN.read_text(encoding="utf-8"))
    payload["surface"]["structure"] = str(REFERENCE_SLAB)
    payload["adsorbates"][0]["file"] = str(REFERENCE_LIGAND)
    payload["export"]["vasp_template"] = str(NOTEBOOK / "inputs" / "vasp_template")
    payload["export"]["output"] = str(tmp_path / "generated")
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _primitive_nio110(tmp_path: Path) -> Path:
    lattice = 4.1863375874188264
    nickel = fcc110("Ni", size=(1, 1, 5), a=lattice, vacuum=None, orthogonal=True, periodic=True)
    oxygen = Atoms(
        ["O"] * len(nickel),
        positions=nickel.positions + nickel.cell[0] / 2,
        cell=nickel.cell,
        pbc=nickel.pbc,
    )
    slab = nickel + oxygen
    slab.set_cell(nickel.cell)
    slab.pbc = (True, True, False)
    slab.center(vacuum=15.0, axis=2)
    path = tmp_path / "primitive.vasp"
    write(path, slab, format="vasp", direct=True, sort=False)
    return path


def test_reference_surface_analysis_and_afm_graph():
    slab = read(REFERENCE_SLAB)
    analysis = analyze_surface(slab, metal="Ni", anion="O")
    assert analysis.n_exposed == 20
    assert set(analysis.coordination[index] for index in analysis.exposed_indices) == {4}

    report = assign_superexchange_afm(slab, magnetic_species="Ni", bridge_species="O")
    assert report["up"] == report["down"] == 50
    assert report["net_moment"] == 0.0
    assert report["exchange_edges"] > 0


def test_reference_campaign_plans_balanced_reactive_grid(tmp_path):
    plan = plan_surface_campaign(_campaign(tmp_path))
    assert plan["state_count"] == 15
    assert plan["decorated_state_count"] == 15
    assert plan["skipped_decorations"] == []
    assert plan["freeze"] == {"frozen_atoms": 40, "total_atoms": 200}
    assert plan["magnetism"]["up"] == plan["magnetism"]["down"] == 50
    assert all(row["initial_geometry_audit"]["minimum_distance_a"] > 0.65 for row in plan["states"])
    graph = plan["state_graph"]
    assert len(graph["nodes"]) == 30
    assert {edge["reaction"] for edge in graph["edges"]} == {
        "increase-coverage",
        "protonate-lattice-oxygen",
        "bind-adsorbate",
    }

    dissociated = [row for row in plan["states"] if row["motif"] == "dissociated_water"]
    assert len(dissociated) == 7
    assert all(row["source_equivalents"]["H2O"] > 0 for row in dissociated)
    assert min(row["periodic_image_gap_a"] for row in plan["decorated_states"]) >= 3.5


def test_surface_campaign_builds_runnable_provenance_stamped_runs(tmp_path):
    result = build_surface_campaign(_campaign(tmp_path))
    assert result["run_count"] == 30
    output = Path(result["output"])
    assert (output / "manifest.csv").is_file()
    assert (output / "state_graph.json").is_file()
    provenance_files = list(output.rglob("provenance.json"))
    assert len(provenance_files) == 30

    decorated = next(path for path in provenance_files if json.loads(path.read_text())["docking"] is not None)
    run = decorated.parent
    provenance = json.loads(decorated.read_text(encoding="utf-8"))
    assert provenance["schema"] == "interfaceforge.reactive-surface/v1"
    assert provenance["docking"]["periodic_image_gap_a"] >= 3.5
    assert provenance["initial_geometry_audit"]["minimum_distance_a"] > 0.65
    assert provenance["frozen_atoms"]
    incar = (run / "INCAR").read_text(encoding="utf-8")
    assert "MAGMOM =" in incar
    assert "ISIF = 2" in incar
    assert "LDIPOL = .TRUE." in incar
    assert not (run / "POTCAR").exists()

    audit = audit_surface_runs(output)
    assert audit["run_count"] == 30
    assert audit["finished"] == 0
    assert all(row["spin_status"] == "MISSING" for row in audit["runs"])
    direct = [row for row in audit["runs"] if row["initial_binding"] == "direct"]
    hbond = [row for row in audit["runs"] if row["initial_binding"] == "hbond"]
    assert direct and all(row["classified_binding"] == "metal-bound" for row in direct)
    assert hbond and all(row["classified_binding"] == "non-chemisorbed" for row in hbond)


def test_cell_optimizer_honors_afm_parity_clearance_and_atom_budget(tmp_path):
    primitive = _primitive_nio110(tmp_path)
    result = optimize_surface_cell(
        primitive,
        adsorbate_path=REFERENCE_LIGAND,
        min_multiplier=12,
        max_multiplier=24,
        max_atoms=240,
        min_translation=14.0,
        min_image_gap=3.5,
        max_aspect=1.30,
        translation_parity=(1, 0),
        output=tmp_path / "best.vasp",
        frozen_bottom_layers=1,
    )
    best = result["best"]
    matrix = np.asarray(best["matrix"], dtype=int)
    assert best["matrix"] == [[4, 0], [0, 5]]
    assert best["atoms"] == 200
    assert best["atoms"] <= 240
    assert best["shortest_translation_a"] >= 14.0
    assert best["adsorbate_image_gap_a"] >= 3.5
    assert np.all(matrix[:, 0] % 2 == 0)
    built = read(tmp_path / "best.vasp")
    assert len(built) == best["atoms"]
    assert built.constraints


def test_surface_selector_preserves_rare_mechanism_groups(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "frame,uncertainty,coverage,motif,reaction_coordinate\n"
        "easy1,1.00,0.0,clean,0.0\n"
        "easy2,0.99,0.0,clean,0.1\n"
        "easy3,0.98,0.0,clean,0.2\n"
        "dissoc1,0.70,0.5,dissociated_water,0.8\n"
        "dissoc2,0.60,0.5,dissociated_water,0.9\n"
        "hbond1,0.50,1.0,hbond,1.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "queue.csv"
    result = select_surface_candidates(
        candidates,
        output,
        count=3,
        feature_columns=("coverage", "reaction_coordinate"),
        max_per_state=1,
    )
    assert result["selected"] == 3
    selected = output.read_text(encoding="utf-8")
    assert "clean" in selected
    assert "dissociated_water" in selected
    assert "hbond" in selected
