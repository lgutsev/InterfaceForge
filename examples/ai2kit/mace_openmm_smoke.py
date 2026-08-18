#!/usr/bin/env python3
"""Compare one custom MACE model through native MACE and OpenMM-ML."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm
from ase.io import read
from mace.calculators import MACECalculator
from openmm import app, unit
from openmmml import MLPotential


def discover_model(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    search_root = path / "mace_model" if (path / "mace_model").is_dir() else path
    candidates = [
        item for item in search_root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".model", ".pt", ".pth"}
    ]
    # MACE training directories may contain both the ordinary stage-two model
    # and an exported/compiled derivative.  Native MACE and OpenMM-ML parity
    # should start from the ordinary model; a compiled file remains usable when
    # the caller supplies its path explicitly.
    ordinary = [item for item in candidates if "compiled" not in item.name.lower()]
    if ordinary:
        candidates = ordinary
    preferred = [
        item for item in candidates
        if any(token in item.name.lower() for token in ("stagetwo", "stage_two", "stage2"))
    ]
    chosen = preferred or candidates
    if len(chosen) != 1:
        listing = "\n".join(f"  {item}" for item in chosen) or "  (none)"
        raise SystemExit(
            f"Expected exactly one stage-two model below {search_root}; found "
            f"{len(chosen)}:\n{listing}\nPass the model file explicitly if needed."
        )
    return chosen[0]


def make_topology(atoms) -> app.Topology:
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for atom in atoms:
        topology.addAtom(atom.symbol, app.Element.getBySymbol(atom.symbol), residue)
    if np.any(atoms.pbc):
        vectors = tuple(openmm.Vec3(*row) * unit.angstrom for row in atoms.cell.array)
        topology.setPeriodicBoxVectors(vectors)
    return topology


def canonicalize_periodic_geometry(atoms):
    """Rotate a periodic ASE structure into OpenMM's box convention."""
    result = atoms.copy()
    if not np.any(result.pbc):
        return result
    if not np.all(result.pbc):
        raise SystemExit(
            "The smoke test requires either three-dimensional periodicity or "
            "a fully non-periodic structure; partial PBC cannot be represented "
            "faithfully by an OpenMM Topology."
        )

    # ASE returns a lower-triangular cell R and an orthogonal transformation Q
    # such that R @ Q equals the input cell.  Retaining fractional coordinates
    # while replacing the cell with R rotates positions and the cell together,
    # preserving the physical structure and satisfying OpenMM's requirement
    # that a is parallel to x and b lies in the xy plane.
    scaled_positions = result.get_scaled_positions(wrap=False)
    standard_cell, _ = result.cell.standard_form()
    result.set_cell(standard_cell, scale_atoms=False)
    result.set_scaled_positions(scaled_positions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model file, mace_model directory, or seed directory")
    parser.add_argument("structure", help="ASE-readable structure (extxyz, POSCAR, etc.)")
    parser.add_argument("--energy-tol", type=float, default=2.0e-3, help="absolute eV tolerance")
    parser.add_argument("--force-tol", type=float, default=2.0e-2, help="max eV/Ang tolerance")
    args = parser.parse_args()

    model = discover_model(args.model)
    atoms = canonicalize_periodic_geometry(read(args.structure, index=0))
    if not openmm.Platform.getNumPlatforms():
        raise SystemExit("OpenMM did not load any compute platform")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; submit this smoke test to a GPU partition")
    print("CUDA:", torch.cuda.get_device_name(0))
    print("Model:", model)
    print("Structure:", Path(args.structure).resolve(), f"({len(atoms)} atoms)")

    native_atoms = atoms.copy()
    native_atoms.calc = MACECalculator(
        model_paths=str(model), device="cuda", default_dtype="float32"
    )
    native_energy = float(native_atoms.get_potential_energy())
    native_forces = np.asarray(native_atoms.get_forces())

    potential = MLPotential("mace", modelPath=str(model))
    topology = make_topology(atoms)
    system = potential.createSystem(topology, precision="single")
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("CUDA")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(np.asarray(atoms.positions) * unit.angstrom)
    state = context.getState(getEnergy=True, getForces=True)
    ml_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 96.4853321233
    ml_forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    ) / 964.853321233

    energy_delta = abs(native_energy - ml_energy)
    force_delta = float(np.max(np.abs(native_forces - ml_forces)))
    print(f"Native MACE energy: {native_energy:.10f} eV")
    print(f"OpenMM-ML energy:    {ml_energy:.10f} eV")
    print(f"Energy |delta|:      {energy_delta:.3e} eV")
    print(f"Force max |delta|:   {force_delta:.3e} eV/Ang")
    if energy_delta > args.energy_tol or force_delta > args.force_tol:
        raise SystemExit("FAIL: native MACE/OpenMM-ML parity exceeds tolerance")
    print("PASS: native MACE and OpenMM-ML agree within tolerance")


if __name__ == "__main__":
    main()
