"""Chemically constrained surface-state generation and molecular docking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    SurfaceAnalysis,
    _ase,
    extend_atoms,
    mic_delta,
    periodic_self_image_gap,
    read_molecule,
    select_sites,
)

CONTACT_MINIMA = {
    tuple(sorted(pair)): value
    for pair, value in {
        ("H", "Ni"): 2.10,
        ("O", "Ni"): 1.75,
        ("O", "O"): 2.30,
        ("H", "O"): 1.45,
        ("C", "Ni"): 1.80,
        ("N", "Ni"): 1.80,
        ("P", "Ni"): 2.00,
    }.items()
}


@dataclass
class ReactiveState:
    name: str
    atoms: Any
    coverage: float
    arrangement: str
    motif: str
    selected_sites: list[int]
    occupied_sites: list[int]
    donor_pairs: list[tuple[int, int]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / norm if norm else np.zeros(3)


def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _unit(a), _unit(b)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.asarray([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / np.dot(cross, cross))


def _hydrogen_direction(index: int, tilt_deg: float = 25.0) -> np.ndarray:
    phi = (index * 137.507764) % 360.0
    tilt = np.radians(tilt_deg)
    azimuth = np.radians(phi)
    return np.asarray([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)])


def _terminal_hydroxyls(atoms, sites: list[int], *, metal_o: float, oh: float):
    positions: list[np.ndarray] = []
    symbols: list[str] = []
    pairs: list[tuple[int, int]] = []
    offset = len(atoms)
    for order, site in enumerate(sites):
        oxygen = atoms.positions[site] + np.asarray([0.0, 0.0, metal_o])
        hydrogen = oxygen + oh * _hydrogen_direction(order)
        o_index = offset + len(symbols)
        positions.extend([oxygen, hydrogen])
        symbols.extend(["O", "H"])
        pairs.append((o_index, o_index + 1))
    extra = _ase()["Atoms"](symbols=symbols, positions=positions) if symbols else _ase()["Atoms"]()
    return extend_atoms(atoms, extra), pairs


def _acceptor_matching(
    atoms,
    analysis: SurfaceAnalysis,
    sites: list[int],
    *,
    minimum: float,
    maximum: float,
    surface_band: float,
) -> dict[int, int]:
    anions = [
        index for index in analysis.anion_indices if atoms.positions[index, 2] >= analysis.exposed_z - surface_band
    ]
    options: dict[int, list[tuple[float, int]]] = {}
    for site in sites:
        choices = []
        for anion in anions:
            distance = float(np.linalg.norm(mic_delta(atoms.positions[anion], atoms.positions[site], atoms.cell)))
            if minimum <= distance <= maximum:
                choices.append((distance, int(anion)))
        options[site] = sorted(choices)

    assigned: dict[int, int] = {}

    def augment(site: int, seen: set[int]) -> bool:
        for _distance, anion in options[site]:
            if anion in seen:
                continue
            seen.add(anion)
            previous = next((owner for owner, value in assigned.items() if value == anion), None)
            if previous is None or augment(previous, seen):
                assigned[site] = anion
                return True
        return False

    for site in sorted(sites, key=lambda item: (len(options[item]), item)):
        augment(site, set())
    missing = [site for site in sites if site not in assigned]
    if missing:
        raise ValueError(
            "dissociated-water state is not stoichiometrically complete: "
            f"no distinct lattice-{analysis.anion} proton acceptor for sites {missing}"
        )
    return assigned


def _initial_geometry_audit(atoms) -> dict[str, Any]:
    positions = atoms.positions
    distances = np.linalg.norm(mic_delta(positions[:, None, :], positions[None, :, :], atoms.cell), axis=-1)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min()) if len(atoms) > 1 else np.inf
    if minimum < 0.65:
        raise ValueError(f"reactive state contains an atomic clash at {minimum:.3f} A")
    symbols = np.asarray(atoms.get_chemical_symbols())
    oxygen = np.where(symbols == "O")[0]
    hydrogen = np.where(symbols == "H")[0]
    unbound_hydrogen = 0
    if len(hydrogen):
        for index in hydrogen:
            nearest = float(np.linalg.norm(mic_delta(positions[oxygen], positions[index], atoms.cell), axis=1).min())
            unbound_hydrogen += int(nearest > 1.25)
    if unbound_hydrogen:
        raise ValueError(f"reactive state has {unbound_hydrogen} H atoms without an O-H bond")
    return {
        "minimum_distance_a": round(minimum, 6),
        "o_h_bonded_hydrogen": int(len(hydrogen) - unbound_hydrogen),
    }


def build_reactive_state(
    base,
    analysis: SurfaceAnalysis,
    *,
    coverage: float,
    arrangement: str,
    motif: str,
    seed: int,
    metal_o_distance: float = 2.0,
    oh_distance: float = 0.97,
    acceptor_minimum: float = 1.5,
    acceptor_maximum: float = 3.2,
    surface_anion_band: float = 1.6,
    name_prefix: str = "surface",
) -> ReactiveState:
    sites = select_sites(base, analysis.exposed_indices, coverage, arrangement, seed)
    percent = int(round(100 * coverage))
    name = name_prefix if not sites else f"{name_prefix}_OH{percent}_{arrangement}_{motif}"
    if not sites:
        clean = base.copy()
        return ReactiveState(
            name,
            clean,
            coverage,
            "clean",
            "clean",
            [],
            [],
            provenance={
                "source_equivalents": {},
                "selected_sites": [],
                "site_count": len(analysis.exposed_indices),
                "coverage_count": 0,
                "initial_geometry_audit": _initial_geometry_audit(clean),
            },
        )

    if motif == "terminal_hydroxyl":
        built, donors = _terminal_hydroxyls(base, sites, metal_o=metal_o_distance, oh=oh_distance)
        source_species = {"O": len(sites), "H": len(sites)}
    elif motif == "dissociated_water":
        matching = _acceptor_matching(
            base,
            analysis,
            sites,
            minimum=acceptor_minimum,
            maximum=acceptor_maximum,
            surface_band=surface_anion_band,
        )
        terminal, donors = _terminal_hydroxyls(base, sites, metal_o=metal_o_distance, oh=oh_distance)
        hydrogen_positions = []
        lattice_pairs: list[tuple[int, int]] = []
        for order, site in enumerate(sites):
            anion = matching[site]
            position = base.positions[anion] + oh_distance * _hydrogen_direction(order + len(sites))
            hydrogen_positions.append(position)
            lattice_pairs.append((anion, len(terminal) + len(hydrogen_positions) - 1))
        protons = _ase()["Atoms"](symbols=["H"] * len(hydrogen_positions), positions=hydrogen_positions)
        built = extend_atoms(terminal, protons)
        donors.extend(lattice_pairs)
        source_species = {"H2O": len(sites)}
    else:
        raise ValueError(f"unknown surface reaction motif {motif!r}")
    return ReactiveState(
        name=name,
        atoms=built,
        coverage=coverage,
        arrangement=arrangement,
        motif=motif,
        selected_sites=sites,
        occupied_sites=sites.copy(),
        donor_pairs=donors,
        provenance={
            "source_equivalents": source_species,
            "selected_sites": sites,
            "site_count": len(analysis.exposed_indices),
            "coverage_count": len(sites),
            "initial_geometry_audit": _initial_geometry_audit(built),
        },
    )


def enumerate_reactive_states(
    base,
    analysis: SurfaceAnalysis,
    *,
    coverages: list[float],
    arrangements: list[str],
    motifs: list[str],
    seed: int,
    parameters: dict[str, Any] | None = None,
    name_prefix: str = "surface",
) -> list[ReactiveState]:
    parameters = parameters or {}
    states: list[ReactiveState] = []
    seen: set[tuple[Any, ...]] = set()
    for coverage in coverages:
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage must lie in [0, 1], got {coverage}")
        if np.isclose(coverage, 0.0):
            key = (0.0, "clean", "clean")
            if key not in seen:
                states.append(
                    build_reactive_state(
                        base,
                        analysis,
                        coverage=0.0,
                        arrangement="clean",
                        motif="clean",
                        seed=seed,
                        name_prefix=name_prefix,
                    )
                )
                seen.add(key)
            continue
        effective_arrangements = ["full"] if np.isclose(coverage, 1.0) else arrangements
        for arrangement in effective_arrangements:
            selection_arrangement = "clustered" if arrangement == "full" else arrangement
            for motif in motifs:
                state = build_reactive_state(
                    base,
                    analysis,
                    coverage=coverage,
                    arrangement=selection_arrangement,
                    motif=motif,
                    seed=seed,
                    name_prefix=name_prefix,
                    **parameters,
                )
                state.arrangement = arrangement
                state.name = state.name.replace(f"_{selection_arrangement}_", f"_{arrangement}_")
                key = (tuple(state.selected_sites), motif)
                if key not in seen:
                    states.append(state)
                    seen.add(key)
    return states


def _phosphonate_anchor(molecule) -> tuple[int, list[int], int, int]:
    symbols = np.asarray(molecule.get_chemical_symbols())
    positions = molecule.positions
    for phosphorus in np.where(symbols == "P")[0]:
        distances = np.linalg.norm(positions - positions[phosphorus], axis=1)
        oxygens = [int(i) for i in np.argsort(distances) if symbols[i] == "O" and distances[i] <= 1.95]
        carbons = [int(i) for i in np.argsort(distances) if symbols[i] == "C" and distances[i] <= 2.10]
        if len(oxygens) == 3 and len(carbons) == 1:
            hydroxyl_oxygens = []
            for oxygen in oxygens:
                h_distance = np.linalg.norm(positions - positions[oxygen], axis=1)
                if any(symbols[i] == "H" and h_distance[i] <= 1.2 for i in range(len(molecule))):
                    hydroxyl_oxygens.append(oxygen)
            terminal = next((oxygen for oxygen in oxygens if oxygen not in hydroxyl_oxygens), oxygens[0])
            return int(phosphorus), oxygens, carbons[0], terminal
    raise ValueError("no phosphonate anchor found (expected P bonded to three O and one C)")


def _orient_phosphonate(molecule, phosphorus: int, oxygens: list[int], carbon: int):
    out = molecule.copy()
    pivot = out.positions[phosphorus].copy()
    body = [i for i in range(len(out)) if i not in {phosphorus, *oxygens}]
    direction = out.positions[body].mean(axis=0) - pivot if body else out.positions[carbon] - pivot
    out.positions = (_rotation(direction, np.asarray([0.0, 0.0, 1.0])) @ (out.positions - pivot).T).T + pivot
    return out


def _contact_margin(surface, molecule, minimum: float) -> float:
    delta = mic_delta(molecule.positions[:, None, :], surface.positions[None, :, :], surface.cell)
    distances = np.linalg.norm(delta, axis=-1)
    ligand_symbols = molecule.get_chemical_symbols()
    surface_symbols = surface.get_chemical_symbols()
    cutoffs = np.asarray(
        [
            [CONTACT_MINIMA.get(tuple(sorted((ligand, substrate))), minimum) for substrate in surface_symbols]
            for ligand in ligand_symbols
        ],
        dtype=float,
    )
    return float((distances - cutoffs).min())


def dock_phosphonate(
    state: ReactiveState,
    analysis: SurfaceAnalysis,
    molecule_path: str | Path,
    *,
    mode: str = "direct",
    metal_o_distance: float = 2.05,
    hbond_distance: float = 1.7,
    minimum_contact: float = 1.4,
    minimum_image_gap: float = 3.5,
    minimum_vacuum_gap: float = 12.0,
    extend_vacuum: bool = True,
):
    """Dock a neutral phosphonic acid by direct metal-O or surface-OH H bonding."""
    molecule = read_molecule(molecule_path)
    phosphorus, oxygens, carbon, binding_oxygen = _phosphonate_anchor(molecule)
    molecule = _orient_phosphonate(molecule, phosphorus, oxygens, carbon)
    if mode == "direct":
        available = sorted(set(analysis.exposed_indices) - set(state.occupied_sites))
        if not available:
            raise ValueError(f"state {state.name} has no unoccupied exposed metal site")
        target_site = available[0]
        target = state.atoms.positions[target_site].copy()
        target[2] += metal_o_distance
        anchor_description = {"mode": mode, "metal_site": target_site}
    elif mode == "hbond":
        if not state.donor_pairs:
            raise ValueError(f"state {state.name} has no surface hydroxyl donor")
        donor_o, donor_h = state.donor_pairs[0]
        direction = _unit(mic_delta(state.atoms.positions[donor_h], state.atoms.positions[donor_o], state.atoms.cell))
        target = state.atoms.positions[donor_h] + hbond_distance * direction
        anchor_description = {"mode": mode, "donor_o": donor_o, "donor_h": donor_h}
    else:
        raise ValueError("phosphonate binding mode must be direct or hbond")

    molecule.positions += target - molecule.positions[binding_oxygen]
    pivot = molecule.positions[binding_oxygen].copy()
    best = molecule.copy()
    best_margin = _contact_margin(state.atoms, best, minimum_contact)
    for angle in np.arange(15.0, 360.0, 15.0):
        radians = np.radians(angle)
        rotation = np.asarray(
            [[np.cos(radians), -np.sin(radians), 0.0], [np.sin(radians), np.cos(radians), 0.0], [0.0, 0.0, 1.0]]
        )
        candidate = molecule.copy()
        candidate.positions = (rotation @ (candidate.positions - pivot).T).T + pivot
        margin = _contact_margin(state.atoms, candidate, minimum_contact)
        if margin > best_margin:
            best, best_margin = candidate, margin
    if best_margin < 0:
        raise ValueError(
            f"no clash-free phosphonate orientation for {state.name}; best contact margin {best_margin:.3f} A"
        )
    image_gap = periodic_self_image_gap(best.positions, state.atoms.cell)
    if image_gap < minimum_image_gap:
        raise ValueError(
            f"phosphonate periodic images approach to {image_gap:.3f} A (minimum {minimum_image_gap:.3f} A)"
        )
    assembled = extend_atoms(state.atoms, best)
    cell = np.asarray(assembled.cell, dtype=float)
    normal_length = float(np.linalg.norm(cell[2]))
    slab_bottom = float(state.atoms.positions[:, 2].min())
    ligand_top = float(best.positions[:, 2].max())
    vacuum_gap = slab_bottom + normal_length - ligand_top
    if vacuum_gap < minimum_vacuum_gap:
        if not extend_vacuum:
            raise ValueError(f"ligand-to-slab-image vacuum is {vacuum_gap:.3f} A (minimum {minimum_vacuum_gap:.3f} A)")
        target_length = normal_length + (minimum_vacuum_gap - vacuum_gap)
        resized = cell.copy()
        resized[2] *= target_length / normal_length
        assembled.set_cell(resized, scale_atoms=False)
        vacuum_gap = minimum_vacuum_gap
    return assembled, {
        **anchor_description,
        "binding_oxygen": binding_oxygen,
        "contact_margin_a": round(best_margin, 6),
        "periodic_image_gap_a": round(image_gap, 6),
        "vacuum_gap_a": round(vacuum_gap, 6),
        "cell_normal_a": round(float(np.linalg.norm(assembled.cell[2])), 6),
        "molecule": str(Path(molecule_path)),
    }
