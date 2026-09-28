"""Synthetic NiO Step1/Step2 VASP trees that ASE's OUTCAR reader can parse.

Only the pieces the dataset exporter reads are written: INCAR, POSCAR (with
selective dynamics), CONTCAR, OSZICAR (MD lines plus electronic iterations),
OUTCAR (header, per-step cell/positions/forces/energies, completion marker)
and ``step2_sample.json``. Geometry and labels are random but reproducible.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CELL = np.diag([8.0, 8.5, 20.0])


def write_incar(path: Path, **tags: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key} = {value}\n" for key, value in tags.items()), encoding="utf-8")


def write_poscar(
    path: Path,
    species: Sequence[str],
    counts: Sequence[int],
    cell: np.ndarray,
    positions: np.ndarray,
    fixed: Sequence[int] = (),
) -> None:
    fixed_set = set(fixed)
    lines = ["synthetic NiO", "1.0"]
    lines += ["  " + " ".join(f"{value:.10f}" for value in row) for row in cell]
    lines += ["  " + " ".join(species), "  " + " ".join(str(count) for count in counts)]
    lines += ["Selective dynamics", "Cartesian"]
    for index, row in enumerate(positions):
        flag = "F F F" if index in fixed_set else "T T T"
        lines.append("  " + " ".join(f"{value:.10f}" for value in row) + f" {flag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def outcar_text(
    species: Sequence[str],
    counts: Sequence[int],
    frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    *,
    complete: bool = True,
    truncate_last: bool = False,
    nan_energy_at: int | None = None,
) -> str:
    lines = [" vasp.6.5.1 synthetic InterfaceForge fixture\n"]
    for symbol in species:
        lines.append(f"   POTCAR:    PAW_PBE {symbol} 02Aug2007\n")
    lines.append("   ISPIN  =      2    spin polarized calculation?\n")
    for symbol in species:
        lines.append(f"   POTCAR:    PAW_PBE {symbol} 02Aug2007\n")
    lines.append("   ions per type =  " + "  ".join(str(count) for count in counts) + "\n")
    for index, (cell, positions, forces, energy) in enumerate(frames):
        lines.append(f"--------------------- Iteration    {index + 1}(   1)  ---------------------\n")
        lines.append("      direct lattice vectors                 reciprocal lattice vectors\n")
        for row in cell:
            lines.append("    " + " ".join(f"{value:13.9f}" for value in row) + "   0.0 0.0 0.0\n")
        lines.append(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        lines.append(" " + "-" * 83 + "\n")
        for position, force in zip(positions, forces, strict=True):
            lines.append(
                "  "
                + " ".join(f"{value:12.6f}" for value in position)
                + "   "
                + " ".join(f"{value:13.6f}" for value in force)
                + "\n"
            )
        lines.append(" " + "-" * 83 + "\n")
        if truncate_last and index == len(frames) - 1:
            break
        value = "NaN" if nan_energy_at == index else f"{energy:.8f}"
        lines.append("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        lines.append("  ---------------------------------------------------\n")
        lines.append(f"  free  energy   TOTEN  =      {energy - 0.001:.8f} eV\n")
        lines.append("\n")
        lines.append(f"  energy  without entropy=      {value}  energy(sigma->0) =      {value}\n")
    if complete and not truncate_last:
        lines.append(" General timing and accounting informations for this job:\n")
    return "".join(lines)


def oszicar_text(temperatures: Sequence[float], energies: Sequence[float], scf: Sequence[int]) -> str:
    lines: list[str] = []
    for step, (temperature, energy, iterations) in enumerate(zip(temperatures, energies, scf, strict=True), start=1):
        for electronic in range(1, iterations + 1):
            lines.append(f"DAV:  {electronic:3d}    -0.100000000000E+04   -0.1E-03   -0.1E-04   100   0.1E-02\n")
        lines.append(
            f"   {step:4d} T=  {temperature:6.1f} E= {energy + 1:.8E} F= {energy:.8E} "
            f"E0= {energy:.8E}  EK= 0.10000E+01 SP= 0.00E+00 SK= 0.00E+00\n"
        )
    return "".join(lines)


def write_run(
    run: Path,
    *,
    species: Sequence[str],
    counts: Sequence[int],
    start_positions: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    tebeg: float,
    teend: float | None = None,
    fixed: Sequence[int] = (0, 1),
    complete: bool = True,
    truncate_last: bool = False,
    scf_ceiling_steps: Sequence[int] = (),
    runaway_from: int | None = None,
    energy_jump_from: int | None = None,
    nan_energy_at: int | None = None,
    nelm: int = 120,
    gz: bool = False,
    write_oszicar: bool = True,
    base_energy: float = -700.0,
) -> np.ndarray:
    """Write one MD run; return its final (CONTCAR) positions."""

    run.mkdir(parents=True, exist_ok=True)
    tags: dict[str, object] = {"IBRION": 0, "NSW": n_steps, "POTIM": 0.5, "TEBEG": tebeg, "NELM": nelm}
    if teend is not None:
        tags["TEEND"] = teend
    write_incar(run / "INCAR", **tags)
    write_poscar(run / "POSCAR", species, counts, CELL, start_positions, fixed)
    natoms = int(sum(counts))
    frames = []
    positions = np.array(start_positions, dtype=float)
    temperatures, energies, scf = [], [], []
    for step in range(n_steps):
        positions = positions + rng.normal(scale=0.01, size=positions.shape)
        forces = rng.normal(scale=0.5, size=(natoms, 3))
        energy = base_energy + 0.01 * rng.normal()
        frames.append((CELL, positions.copy(), forces, energy))
        target = teend if teend is not None else tebeg
        temperature = target + rng.normal(scale=10.0)
        if runaway_from is not None and step + 1 >= runaway_from:
            temperature = 5000.0
        temperatures.append(temperature)
        jump = 100.0 if energy_jump_from is not None and step + 1 >= energy_jump_from else 0.0
        energies.append(energy + jump)
        scf.append(nelm if step + 1 in set(scf_ceiling_steps) else 12)
    text = outcar_text(
        species, counts, frames, complete=complete, truncate_last=truncate_last, nan_energy_at=nan_energy_at
    )
    if gz:
        with gzip.open(run / "OUTCAR.gz", "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        (run / "OUTCAR").write_text(text, encoding="utf-8")
    if write_oszicar:
        (run / "OSZICAR").write_text(oszicar_text(temperatures, energies, scf), encoding="utf-8")
    write_poscar(run / "CONTCAR", species, counts, CELL, positions, fixed)
    return positions


NIO_CASES = (
    ("OH0", "NiO_m110_Big_U46", ("Ni", "O"), (4, 4)),
    ("OH50", "NiO_m110_Big_U46_OH50_clustered_capped", ("H", "Ni", "O"), (2, 4, 6)),
    ("OH50", "NiO_m110_Big_U46_OH50_clustered_dissoc", ("H", "Ni", "O"), (4, 4, 6)),
    (
        "OH50",
        "NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary",
        ("C", "H", "N", "Ni", "O", "P"),
        (2, 3, 1, 4, 8, 1),
    ),
    ("OH25", "NiO_m110_Big_U46_OH25_scattered_dissoc_DCZ4P_bare", ("C", "H", "N", "Ni", "O", "P"), (2, 3, 1, 4, 7, 1)),
    ("OH75", "NiO_m110_Big_U46_OH75_scattered_capped", ("H", "Ni", "O"), (3, 4, 7)),
)


def build_nio_tree(
    head: Path,
    *,
    cases: Sequence[tuple[str, str, Sequence[str], Sequence[int]]] = NIO_CASES,
    temperatures: Sequence[int] = (300, 450, 600),
    step1_steps: int = 6,
    step2_steps: int = 8,
    seed: int = 7,
    sample_every: int = 2,
    special: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Build ``head/Step1/<OH>/<case>`` and ``head/Step2_<T>K/<OH>/<case>`` trees.

    Step2 POSCAR is the Step1 CONTCAR (the real ``step2-prepare`` hand-off), and
    each Step2 root gets a ``step2_sample.json`` selecting every
    ``sample_every``-th frame. ``special`` maps "<stage>/<case>" to extra
    ``write_run`` keyword arguments (truncation, SCF ceiling, runaway ...).
    """

    rng = np.random.default_rng(seed)
    special = dict(special or {})
    for group, case, species, counts in cases:
        natoms = int(sum(counts))
        start = rng.random((natoms, 3)) * np.array([7.5, 8.0, 6.0]) + np.array([0.2, 0.2, 5.0])
        step1 = head / "Step1" / group / case
        extra = dict(special.get(f"Step1/{case}", {}))
        final = write_run(
            step1,
            species=species,
            counts=counts,
            start_positions=start,
            n_steps=step1_steps,
            rng=rng,
            tebeg=100.0,
            teend=float(temperatures[0]),
            **extra,
        )
        for temperature in temperatures:
            run = head / f"Step2_{temperature}K" / group / case
            extra = dict(special.get(f"Step2_{temperature}K/{case}", {}))
            write_run(
                run,
                species=species,
                counts=counts,
                start_positions=final,
                n_steps=step2_steps,
                rng=rng,
                tebeg=float(temperature),
                **extra,
            )
    for temperature in temperatures:
        root = head / f"Step2_{temperature}K"
        runs = []
        for group, case, _, _ in cases:
            runs.append(
                {
                    "status": "OK",
                    "relative_path": f"{group}/{case}",
                    "indices": list(range(0, step2_steps, sample_every)),
                    "kept_frames": len(range(0, step2_steps, sample_every)),
                }
            )
        (root / "step2_sample.json").write_text(
            json.dumps({"format": "interfaceforge-step2-sample", "schema_version": 1, "runs": runs}, indent=2),
            encoding="utf-8",
        )
    return head
