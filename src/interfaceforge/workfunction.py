"""Planar-averaged LOCPOT analysis and work-function estimation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError


def _fermi_energy(outcar: Path) -> float:
    text = outcar.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"E-fermi\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?)", text)
    if not matches:
        raise SafetyError(f"No E-fermi value found in {outcar}")
    return float(matches[-1])


def analyze_workfunction(
    locpot: str | Path,
    outcar: str | Path,
    *,
    axis: str = "z",
    data_output: str | Path = "locpot.dat",
    plot_output: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate a planar average and report the maximum vacuum potential."""

    try:
        from ase.calculators.vasp import VaspChargeDensity
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for LOCPOT analysis. Install interfaceforge[vasp]."
        ) from exc
    input_path = Path(locpot).resolve()
    outcar_path = Path(outcar).resolve()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    density = VaspChargeDensity(str(input_path))
    cell = density.atoms[0].cell
    lengths = np.linalg.norm(cell, axis=1)
    volume = float(np.linalg.det(cell))
    average_axes = tuple(index for index in (0, 1, 2) if index != axis_index)
    potential = np.mean(density.chg[0], axis=average_axes) * volume
    distance = np.linspace(0, lengths[axis_index], potential.shape[0], endpoint=False)
    fermi = _fermi_energy(outcar_path)
    shifted = potential - fermi
    vacuum = float(np.max(potential))
    work_function = vacuum - fermi
    maximum_index = int(np.argmax(potential))

    data_path = Path(data_output).resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_path,
        np.c_[distance, potential, shifted],
        fmt="%15.8f",
        header="distance_A planar_potential_eV potential_minus_Efermi_eV",
    )
    if plot_output:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise DependencyError("matplotlib is required for --plot-output") from exc
        figure, axes = plt.subplots(figsize=(8, 4.5))
        axes.plot(distance, shifted, color="#145c66", linewidth=1.5)
        axes.axhline(work_function, color="#b23a48", linestyle="--", label=f"vacuum − EF = {work_function:.3f} eV")
        axes.set(xlabel=f"Distance along {axis} (Å)", ylabel="Potential − EF (eV)")
        axes.grid(alpha=0.25)
        axes.legend()
        figure.tight_layout()
        figure.savefig(Path(plot_output).resolve(), dpi=300)
        plt.close(figure)

    summary = {
        "locpot": str(input_path),
        "outcar": str(outcar_path),
        "axis": axis,
        "fermi_energy_ev": fermi,
        "vacuum_potential_ev": vacuum,
        "work_function_ev": work_function,
        "vacuum_position_a": float(distance[maximum_index]),
        "data": str(data_path),
        "plot": str(Path(plot_output).resolve()) if plot_output else None,
        "caution": (
            "The maximum planar potential is an estimator. Inspect the curve for a "
            "flat vacuum plateau and use dipole corrections where appropriate."
        ),
    }
    summary_path = (
        Path(summary_output).resolve()
        if summary_output
        else data_path.with_name("workfunction_summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
