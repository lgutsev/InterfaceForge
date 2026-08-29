from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "nio_m110_hydroxylation"
UTILS_PATH = NOTEBOOK_DIR / "nio_hydroxylation_utils.py"


def load_utils():
    spec = importlib.util.spec_from_file_location("nio_hydroxylation_utils", UTILS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def utils():
    return load_utils()


def test_linear_hbond_is_reported_as_satisfied(utils):
    donor = utils.Hydroxyl(
        o_pos=np.array([2.0, 2.0, 2.0]),
        kind="ni_oh",
        h_pos=np.array([2.97, 2.0, 2.0]),
    )
    acceptor = utils.Hydroxyl(o_pos=np.array([4.8, 2.0, 2.0]), kind="lattice_oh")

    diagnostic = utils.hbond_diagnostic([donor, acceptor], np.diag([10.0, 10.0, 10.0]))

    assert diagnostic.candidate_pairs == 1
    assert diagnostic.satisfied_contacts == 1
    assert diagnostic.score == 1.0


def synthetic_surface(utils, *, include_second_oxygen: bool = True):
    symbols = ["Ni", "Ni", "O"] + (["O"] if include_second_oxygen else [])
    positions = [[0.0, 0.0, 5.0], [3.0, 0.0, 5.0], [1.5, 0.0, 5.0]]
    if include_second_oxygen:
        positions.append([-2.0, 0.0, 5.0])
    atoms = Atoms(symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    return utils.SurfaceModel(
        atoms=atoms,
        ni_indices=np.array([0, 1]),
        o_indices=np.arange(2, len(atoms)),
        exposed_ni=np.array([0, 1]),
        top_z=5.0,
    )


def test_dissociated_pair_uses_complete_distinct_matching(utils):
    hydroxyls = utils.build_dissociated_pair(synthetic_surface(utils), [0, 1])
    lattice_oxygen = [h.parent_o for h in hydroxyls if h.kind == "lattice_oh"]

    assert len([h for h in hydroxyls if h.kind == "ni_oh"]) == 2
    assert sorted(lattice_oxygen) == [2, 3]


def test_dissociated_pair_rejects_incomplete_water_stoichiometry(utils):
    with pytest.raises(ValueError, match="incomplete dissociated-water motif"):
        utils.build_dissociated_pair(
            synthetic_surface(utils, include_second_oxygen=False), [0, 1]
        )


def test_full_real_surface_has_one_lattice_proton_per_selected_ni(utils):
    slab = utils.load_structure(NOTEBOOK_DIR / "inputs" / "CONTCAR").repeat((2, 2, 1))
    surface = utils.analyse_surface(slab)
    hydroxyls = utils.build_dissociated_pair(surface, surface.exposed_ni)

    assert len(surface.exposed_ni) == 48
    assert sum(h.kind == "ni_oh" for h in hydroxyls) == 48
    assert sum(h.kind == "lattice_oh" for h in hydroxyls) == 48
    assert len({h.parent_o for h in hydroxyls if h.kind == "lattice_oh"}) == 48


def test_ligand_binding_oxygen_is_placed_over_target_ni(utils):
    slab = Atoms(
        ["Ni", "O"],
        positions=[[5.0, 5.0, 5.0], [3.0, 5.0, 5.0]],
        cell=[15.0, 15.0, 20.0],
        pbc=True,
    )
    molecule = Atoms(
        ["P", "O", "O", "O", "C"],
        positions=[
            [0.0, 0.0, 1.5],
            [0.3, 0.0, 0.0],
            [-1.0, 0.0, 1.5],
            [1.0, 0.0, 1.5],
            [0.0, 0.0, 3.0],
        ],
    )
    anchor = utils.PhosphonateAnchor(0, [1, 2, 3], [2, 3], [1], 4, [0])

    structure, lift = utils.place_ligand(
        slab,
        molecule,
        anchor,
        [5.0, 5.0],
        ni_plane_z=5.0,
        oh_height=2.05,
        anchor_o_index=1,
    )

    binding_o = structure.positions[len(slab) + 1]
    assert lift == 0.0
    assert np.allclose(binding_o, [5.0, 5.0, 7.05])
    assert np.linalg.norm(binding_o - slab.positions[0]) == pytest.approx(2.05)


def test_steric_search_keeps_binding_oxygen_fixed(utils):
    slab = Atoms(
        ["Ni", "O"],
        positions=[[5.0, 5.0, 5.0], [7.0, 5.0, 7.05]],
        cell=[15.0, 15.0, 20.0],
        pbc=True,
    )
    molecule = Atoms(
        ["P", "O", "O", "O", "C"],
        positions=[
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # initially coincides with the second slab atom
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 3.0],
        ],
    )
    anchor = utils.PhosphonateAnchor(0, [1, 2, 3], [2, 3], [1], 4, [0])

    structure, contact_tilt = utils.place_ligand(
        slab,
        molecule,
        anchor,
        [5.0, 5.0],
        ni_plane_z=5.0,
        anchor_o_index=1,
    )

    ligand = structure.positions[len(slab) :]
    assert contact_tilt == 0.0  # an azimuth adjustment is sufficient
    assert np.allclose(ligand[1], [5.0, 5.0, 7.05])
    assert utils._min_gap(ligand, slab.positions, slab.cell) >= 1.25


def test_slab_incar_overrides_never_relax_vacuum_cell(utils):
    slab = Atoms("NiO", positions=[[0, 0, 4], [0, 0, 6]], cell=[10, 10, 20], pbc=True)

    clean = utils.slab_incar_overrides(slab, decorated=False)
    decorated = utils.slab_incar_overrides(slab, decorated=True)

    assert clean == {"ISIF": "2", "LDIPOL": ".FALSE."}
    assert decorated["ISIF"] == "2"
    assert decorated["LDIPOL"] == ".TRUE."
    assert decorated["IDIPOL"] == "3"
    assert decorated["DIPOL"].startswith("0.5 0.5 ")


def test_run_input_export_requires_magnetic_ni_and_never_writes_potcar(utils, tmp_path):
    template = tmp_path / "template"
    case = tmp_path / "case"
    template.mkdir()
    case.mkdir()
    (template / "INCAR").write_text(
        "SYSTEM = old\nMAGMOM = 2*2 2*-2\nLDAUL = 2 -1\n"
        "LDAUU = 4.6 0\nLDAUJ = 0 0\nISIF = 2\n",
        encoding="utf-8",
    )
    (template / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n", encoding="utf-8")
    structure = Atoms(
        ["H", "Ni", "Ni", "O"],
        positions=np.zeros((4, 3)),
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    with pytest.raises(ValueError, match="without explicit initial magnetic moments"):
        utils.write_run_inputs(case, structure, template)

    structure.set_initial_magnetic_moments([0.0, 2.0, -2.0, 0.0])
    notes = utils.write_run_inputs(case, structure, template, system_name="safe")
    incar = (case / "INCAR").read_text(encoding="utf-8")
    assert "SYSTEM = safe" in incar
    assert "MAGMOM = 1*0 1*2 1*-2 1*0" in incar
    assert "LDAUL = -1 2 -1" in incar
    assert notes["species_order"] == ["H", "Ni", "O"]
    assert not (case / "POTCAR").exists()


def test_notebook_toolkit_is_synchronized():
    utils_tree = ast.parse(UTILS_PATH.read_text(encoding="utf-8"))
    notebook = json.loads(
        (NOTEBOOK_DIR / "NiO_m110_hydroxylation.ipynb").read_text(encoding="utf-8")
    )
    notebook_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    notebook_tree = ast.parse(notebook_code)
    wanted = {
        node.name: ast.dump(node, include_attributes=False)
        for node in utils_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    inlined = {
        node.name: ast.dump(node, include_attributes=False)
        for node in notebook_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    assert {name: inlined.get(name) for name in wanted} == wanted
