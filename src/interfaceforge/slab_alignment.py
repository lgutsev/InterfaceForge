"""Auditable vacuum alignment for families of asymmetric VASP slabs."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError

ATOMIC_MASS = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "Sr": 87.62,
    "Zr": 91.224,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "I": 126.904,
    "Cs": 132.905,
    "Ba": 137.33,
    "Pt": 195.08,
    "Au": 196.97,
    "Pb": 207.2,
    "Bi": 208.98,
}

OUTPUT_FIELDS = [
    "folder",
    "reference",
    "status",
    "error",
    "flatness_status",
    "flat_enough",
    "audit_action",
    "relaunch_review_required",
    "dipole_fix_path",
    "review_flag_path",
    "band_edge_status",
    "selected_side",
    "efermi_eV",
    "vacuum_eV",
    "vacuum_minus_ef_eV",
    "vbm_eV",
    "cbm_eV",
    "gap_eV",
    "vbm_vac_eV",
    "cbm_vac_eV",
    "delta_vbm_eV",
    "delta_cbm_eV",
    "pdos_review_required",
    "selected_slope_eV_per_A",
    "selected_swing_eV",
    "selected_std_eV",
    "selected_correction_step_detected",
    "selected_correction_step_A",
    "selected_correction_step_eV",
    "selected_correction_step_width_A",
    "low_vacuum_eV",
    "high_vacuum_eV",
    "high_minus_low_vacuum_eV",
    "low_slope_eV_per_A",
    "high_slope_eV_per_A",
    "suggested_DIPOL_z",
    "compactness_R",
    "current_LDIPOL",
    "current_IDIPOL",
    "current_DIPOL",
    "sumo_status",
]

AUDIT_FIELDS = [
    "folder",
    "flatness_status",
    "flat_enough",
    "selected_side",
    "selected_slope_eV_per_A",
    "selected_swing_eV",
    "selected_std_eV",
    "selected_correction_step_detected",
    "selected_correction_step_A",
    "selected_correction_step_eV",
    "selected_correction_step_width_A",
    "suggested_DIPOL_z",
    "compactness_R",
    "current_LDIPOL",
    "current_IDIPOL",
    "current_DIPOL",
    "audit_action",
    "relaunch_review_required",
    "dipole_fix_path",
    "review_flag_path",
    "error",
]

OK_MARKER = "LOCPOT_FLATNESS_OK"
REVIEW_MARKER = "RELAUNCH_REVIEW_REQUIRED"
AUDIT_FAILED_MARKER = "LOCPOT_AUDIT_FAILED"


@dataclass
class Structure:
    cell: np.ndarray
    species: list[str]
    counts: list[int]
    fractional: np.ndarray
    coordinate_end_line: int

    @property
    def c_length(self) -> float:
        return float(np.linalg.norm(self.cell[2]))

    @property
    def z_angstrom(self) -> np.ndarray:
        return np.mod(self.fractional[:, 2], 1.0) * self.c_length

    @property
    def elements(self) -> list[str]:
        values: list[str] = []
        for symbol, count in zip(self.species, self.counts, strict=True):
            values.extend([symbol] * count)
        return values


@dataclass
class Plateau:
    side: str
    plateau_eV: float
    slope_eV_per_A: float
    swing_eV: float
    residual_std_eV: float
    window_start_A: float
    window_end_A: float
    window_width_A: float
    npoints: int
    correction_step_detected: bool = False
    correction_step_A: float | None = None
    correction_step_eV: float | None = None
    correction_step_width_A: float | None = None


@dataclass
class ProfileAnalysis:
    cut_A: float
    c_length_A: float
    atom_low_A: float
    atom_high_A: float
    low: Plateau
    high: Plateau


def _next_nonempty(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _all_integers(tokens: list[str]) -> bool:
    try:
        [int(token) for token in tokens]
    except ValueError:
        return False
    return True


def parse_poscar_lines(lines: list[str]) -> Structure:
    """Parse the structure header shared by POSCAR and LOCPOT."""

    if len(lines) < 8:
        raise SafetyError("VASP structure header is too short")
    scale_values = [float(value) for value in lines[1].split()]
    raw_cell = np.array([[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)])
    if len(scale_values) == 1:
        scale = scale_values[0]
        if scale < 0:
            scale = (abs(scale) / abs(np.linalg.det(raw_cell))) ** (1.0 / 3.0)
        cell = raw_cell * scale
        cart_scale = np.array([scale, scale, scale])
    elif len(scale_values) == 3:
        cart_scale = np.array(scale_values)
        cell = raw_cell * cart_scale[np.newaxis, :]
    else:
        raise SafetyError("Unsupported POSCAR scale line")

    line5 = lines[5].split()
    if _all_integers(line5):
        species = [f"X{index + 1}" for index in range(len(line5))]
        counts = [int(value) for value in line5]
        counts_index = 5
    else:
        species = line5
        counts_index = _next_nonempty(lines, 6)
        counts = [int(value) for value in lines[counts_index].split()]
    if len(species) != len(counts):
        raise SafetyError("Species/count mismatch in VASP structure")

    mode_index = _next_nonempty(lines, counts_index + 1)
    if lines[mode_index].strip().lower().startswith("s"):
        mode_index = _next_nonempty(lines, mode_index + 1)
    mode = lines[mode_index].strip().lower()
    start = _next_nonempty(lines, mode_index + 1)
    coordinates: list[list[float]] = []
    index = start
    while len(coordinates) < sum(counts) and index < len(lines):
        if lines[index].strip():
            fields = lines[index].split()
            coordinates.append([float(fields[0]), float(fields[1]), float(fields[2])])
        index += 1
    if len(coordinates) != sum(counts):
        raise SafetyError("Fewer coordinates than declared atoms")
    coordinate_array = np.array(coordinates)
    if mode.startswith("d"):
        fractional = coordinate_array
    elif mode.startswith(("c", "k")):
        fractional = (coordinate_array * cart_scale) @ np.linalg.inv(cell)
    else:
        raise SafetyError(f"Unknown coordinate mode: {lines[mode_index]}")
    return Structure(cell, species, counts, fractional, index)


def read_locpot(path: str | Path) -> tuple[Structure, np.ndarray, np.ndarray]:
    """Read the raw LOCPOT and return its z-planar average in eV."""

    input_path = Path(path)
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    structure = parse_poscar_lines(lines)
    grid_index = _next_nonempty(lines, structure.coordinate_end_line)
    try:
        grid = [int(value) for value in lines[grid_index].split()[:3]]
    except (IndexError, ValueError) as exc:
        raise SafetyError(f"LOCPOT grid dimensions not found in {input_path}") from exc
    if len(grid) != 3:
        raise SafetyError(f"LOCPOT grid dimensions not found in {input_path}")
    nx, ny, nz = grid
    required = nx * ny * nz
    values = np.fromstring(" ".join(lines[grid_index + 1 :]), sep=" ", count=required)
    if values.size != required:
        raise SafetyError(f"{input_path} has {values.size} grid values; expected {required}")
    # VASP writes x fastest, then y, then z. LOCPOT is already in eV and must
    # not receive the volume rescaling that ASE applies to charge densities.
    potential = values.reshape((nz, ny, nx)).mean(axis=(1, 2))
    z_grid = np.arange(nz, dtype=float) * structure.c_length / nz
    return structure, z_grid, potential


def efermi_from_outcar(path: str | Path) -> float:
    matches = re.findall(
        r"E-fermi\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        Path(path).read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        raise SafetyError(f"E-fermi not found in {path}")
    return float(matches[-1])


def band_edges_from_vasprun(path: str | Path) -> tuple[float, float, float, float]:
    root = ET.parse(path).getroot()
    efermi_values = [float(node.text.split()[0]) for node in root.findall(".//i[@name='efermi']") if node.text]
    if not efermi_values:
        raise SafetyError(f"E-fermi not found in {path}")
    blocks = root.findall(".//eigenvalues")
    if not blocks:
        raise SafetyError(f"Eigenvalues not found in {path}")
    energies: list[float] = []
    occupancies: list[float] = []
    for record in blocks[-1].findall(".//r"):
        if record.text:
            fields = record.text.split()
            if len(fields) >= 2:
                energies.append(float(fields[0]))
                occupancies.append(float(fields[1]))
    if not energies:
        raise SafetyError(f"Final eigenvalue block is empty in {path}")
    energy = np.array(energies)
    occupancy = np.array(occupancies)
    maximum = float(np.max(occupancy))
    if maximum <= 0:
        raise SafetyError(f"All eigenvalue occupations are zero in {path}")
    occupied = occupancy > 0.5 * maximum
    if not occupied.any() or occupied.all():
        raise SafetyError(f"Could not separate occupied and unoccupied bands in {path}")
    vbm = float(np.max(energy[occupied]))
    cbm = float(np.min(energy[~occupied]))
    return float(efermi_values[-1]), vbm, cbm, max(0.0, cbm - vbm)


def largest_periodic_gap(z_values: np.ndarray, length: float) -> tuple[float, float, float]:
    values = np.sort(np.unique(np.mod(z_values, length)))
    if values.size < 2:
        raise SafetyError("At least two distinct z coordinates are required")
    next_values = np.r_[values[1:], values[0] + length]
    gaps = next_values - values
    index = int(np.argmax(gaps))
    return float(values[index]), float(next_values[index]), float(gaps[index])


def _localized_steps(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    minimum_delta_eV: float,
    outlier_factor: float,
    maximum_width_A: float,
) -> list[tuple[float, float, float, float]]:
    """Return short potential resets while ignoring a distributed field.

    A dipole-correction reset has a large derivative only over a compact
    interval, whereas a residual electric field has a similar derivative
    throughout the vacuum.  Each result is ``(start, end, center, delta)``.
    """

    edge_widths = np.diff(x_values)
    if edge_widths.size < 3 or np.any(edge_widths <= 0):
        return []
    edge_slopes = np.diff(y_values) / edge_widths
    baseline_slope = float(np.median(edge_slopes))
    slope_deviation = np.abs(edge_slopes - baseline_slope)
    slope_mad = float(np.median(slope_deviation))
    gradient_floor = minimum_delta_eV / maximum_width_A
    threshold = max(gradient_floor, outlier_factor * slope_mad)
    active = slope_deviation >= threshold

    runs: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            runs.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        runs.append((start_index, len(active) - 1))

    steps: list[tuple[float, float, float, float]] = []
    context_A = min(1.0, maximum_width_A)
    for first_edge, last_edge in runs:
        transition_start = float(x_values[first_edge])
        transition_end = float(x_values[last_edge + 1])
        if transition_end - transition_start > maximum_width_A:
            continue
        left = (x_values < transition_start) & (x_values >= transition_start - context_A)
        right = (x_values > transition_end) & (x_values <= transition_end + context_A)
        if np.count_nonzero(left) < 3 or np.count_nonzero(right) < 3:
            continue
        delta = float(np.median(y_values[right]) - np.median(y_values[left]))
        if abs(delta) < minimum_delta_eV:
            continue
        steps.append(
            (
                transition_start,
                transition_end,
                0.5 * (transition_start + transition_end),
                delta,
            )
        )
    return steps


def _fit_plateau(
    side: str,
    shifted_z: np.ndarray,
    potential: np.ndarray,
    start: float,
    end: float,
    minimum_window_angstrom: float,
    discontinuity_min_eV: float,
    discontinuity_factor: float,
    discontinuity_max_width_angstrom: float,
    discontinuity_margin_angstrom: float,
) -> Plateau:
    mask = (shifted_z >= start) & (shifted_z <= end)
    if np.count_nonzero(mask) < 5:
        raise SafetyError(f"{side} vacuum window contains fewer than five points")
    x_values = shifted_z[mask]
    y_values = potential[mask]
    order = np.argsort(x_values)
    x_values, y_values = x_values[order], y_values[order]

    # LDIPOL adds a sawtooth-like reset that can span several grid points. It
    # is a feature of the corrected periodic potential, not a residual field.
    # Detect compact derivative excursions rather than requiring one large
    # point-to-point jump. A gradual slope is retained and fails normally.
    steps = _localized_steps(
        x_values,
        y_values,
        minimum_delta_eV=discontinuity_min_eV,
        outlier_factor=discontinuity_factor,
        maximum_width_A=discontinuity_max_width_angstrom,
    )
    if side == "low-z":
        steps = [
            step
            for step in steps
            if end - step[1] >= minimum_window_angstrom + discontinuity_margin_angstrom
        ]
    else:
        steps = [
            step
            for step in steps
            if step[0] - start >= minimum_window_angstrom + discontinuity_margin_angstrom
        ]
    correction_step_detected = bool(steps)
    correction_step_A: float | None = None
    correction_step_eV: float | None = None
    correction_step_width_A: float | None = None
    if correction_step_detected:
        transition_start, transition_end, correction_step_A, correction_step_eV = (
            steps[-1] if side == "low-z" else steps[0]
        )
        correction_step_width_A = transition_end - transition_start
        if side == "low-z":
            keep = x_values >= transition_end + discontinuity_margin_angstrom
        else:
            keep = x_values <= transition_start - discontinuity_margin_angstrom
        if np.count_nonzero(keep) < 5:
            raise SafetyError(
                f"{side} surface-adjacent plateau has fewer than five points after excluding the dipole step"
            )
        x_values, y_values = x_values[keep], y_values[keep]

    width = float(x_values[-1] - x_values[0])
    if width < minimum_window_angstrom:
        raise SafetyError(
            f"{side} surface-adjacent plateau is only {width:.3f} A after excluding the dipole step"
        )
    slope, intercept = np.polyfit(x_values, y_values, 1)
    residual = y_values - (slope * x_values + intercept)
    return Plateau(
        side=side,
        plateau_eV=float(np.median(y_values)),
        slope_eV_per_A=float(slope),
        swing_eV=float(abs(slope) * width),
        residual_std_eV=float(np.std(residual)),
        window_start_A=float(x_values[0]),
        window_end_A=float(x_values[-1]),
        window_width_A=width,
        npoints=int(x_values.size),
        correction_step_detected=correction_step_detected,
        correction_step_A=correction_step_A,
        correction_step_eV=correction_step_eV,
        correction_step_width_A=correction_step_width_A,
    )


def analyze_profile(
    structure: Structure,
    z_grid: np.ndarray,
    potential: np.ndarray,
    buffer_angstrom: float = 2.0,
    minimum_window_angstrom: float = 2.0,
    discontinuity_min_eV: float = 0.05,
    discontinuity_factor: float = 10.0,
    discontinuity_max_width_angstrom: float = 1.5,
    discontinuity_margin_angstrom: float = 0.5,
) -> tuple[ProfileAnalysis, np.ndarray, np.ndarray]:
    """Fit the two physical vacuum sides independently across a periodic cell."""

    length = structure.c_length
    gap_start, _gap_end, gap_width = largest_periodic_gap(structure.z_angstrom, length)
    if gap_width < 2 * (buffer_angstrom + minimum_window_angstrom):
        raise SafetyError(f"Total periodic vacuum gap {gap_width:.3f} A is too small to analyze both sides")
    cut = (gap_start + 0.5 * gap_width) % length
    atom_shifted = np.mod(structure.z_angstrom - cut, length)
    grid_shifted = np.mod(z_grid - cut, length)
    order = np.argsort(grid_shifted)
    grid_shifted, potential_shifted = grid_shifted[order], potential[order]
    atom_low = float(np.min(atom_shifted))
    atom_high = float(np.max(atom_shifted))
    low_start, low_end = buffer_angstrom, atom_low - buffer_angstrom
    high_start, high_end = atom_high + buffer_angstrom, length - buffer_angstrom
    if low_end - low_start < minimum_window_angstrom:
        raise SafetyError(f"Low-z vacuum window is only {low_end - low_start:.3f} A")
    if high_end - high_start < minimum_window_angstrom:
        raise SafetyError(f"High-z vacuum window is only {high_end - high_start:.3f} A")
    fit_options = {
        "minimum_window_angstrom": minimum_window_angstrom,
        "discontinuity_min_eV": discontinuity_min_eV,
        "discontinuity_factor": discontinuity_factor,
        "discontinuity_max_width_angstrom": discontinuity_max_width_angstrom,
        "discontinuity_margin_angstrom": discontinuity_margin_angstrom,
    }
    low = _fit_plateau("low-z", grid_shifted, potential_shifted, low_start, low_end, **fit_options)
    high = _fit_plateau("high-z", grid_shifted, potential_shifted, high_start, high_end, **fit_options)
    return (
        ProfileAnalysis(cut, length, atom_low, atom_high, low, high),
        grid_shifted,
        potential_shifted,
    )


def ionic_center_fraction(structure: Structure) -> tuple[float, float, list[str]]:
    """Return a periodic mass-weighted center suitable for VASP's DIPOL tag."""

    z_fraction = np.mod(structure.fractional[:, 2], 1.0)
    masses: list[float] = []
    missing: list[str] = []
    for element in structure.elements:
        mass = ATOMIC_MASS.get(element)
        if mass is None:
            missing.append(element)
            mass = 1.0
        masses.append(mass)
    weights = np.array(masses, dtype=float)
    angles = 2 * np.pi * z_fraction
    cosine = np.sum(weights * np.cos(angles)) / np.sum(weights)
    sine = np.sum(weights * np.sin(angles)) / np.sum(weights)
    center = float((np.arctan2(sine, cosine) % (2 * np.pi)) / (2 * np.pi))
    compactness = float(np.hypot(cosine, sine))
    return center, compactness, sorted(set(missing))


def parse_incar(path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {"LDIPOL": None, "IDIPOL": None, "DIPOL": None}
    input_path = Path(path)
    if not input_path.is_file():
        return result
    for raw in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        key = key.upper()
        if key == "LDIPOL":
            result[key] = value.strip(". ").upper().startswith("T")
        elif key == "IDIPOL":
            result[key] = int(value.split()[0])
        elif key == "DIPOL":
            result[key] = [float(item) for item in value.split()[:3]]
    return result


def write_dipole_preview(calc_dir: str | Path, suggested_z: float) -> Path:
    """Write INCAR.dipole_fix without modifying the calculation's INCAR."""

    directory = Path(calc_dir)
    source = directory / "INCAR"
    lines = source.read_text(encoding="utf-8").splitlines() if source.is_file() else []
    current = parse_incar(source).get("DIPOL")
    x_value = current[0] if current and len(current) >= 3 else 0.5
    y_value = current[1] if current and len(current) >= 3 else 0.5
    replacements = {
        "LDIPOL": "LDIPOL = .TRUE.",
        "IDIPOL": "IDIPOL = 3",
        "DIPOL": f"DIPOL  = {x_value:.6f} {y_value:.6f} {suggested_z:.6f}",
    }
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        match = re.match(r"^\s*(LDIPOL|IDIPOL|DIPOL)\s*=", line, flags=re.I)
        if match:
            key = match.group(1).upper()
            output.append(replacements[key])
            seen.add(key)
        else:
            output.append(line)
    if seen != set(replacements):
        output.extend(["", "# Dipole correction preview generated by InterfaceForge"])
        for key in ("LDIPOL", "IDIPOL", "DIPOL"):
            if key not in seen:
                output.append(replacements[key])
    destination = directory / "INCAR.dipole_fix"
    destination.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return destination


def load_alignment_config(path: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "side": "high-z",
        "buffer_angstrom": 2.0,
        "minimum_window_angstrom": 2.0,
        "discontinuity_min_eV": 0.05,
        "discontinuity_factor": 10.0,
        "discontinuity_max_width_angstrom": 1.5,
        "discontinuity_margin_angstrom": 0.5,
        "swing_warn_eV": 0.03,
        "swing_fail_eV": 0.10,
        "std_warn_eV": 0.02,
        "std_fail_eV": 0.05,
        "references": [
            {"prefix": "MAPI_MAI_Surf", "reference": "MAPI_MAI_Surf"},
            {"prefix": "MAPI_PbI2_Surf", "reference": "MAPI_PbI2_Surf"},
        ],
    }
    input_path = Path(path)
    if input_path.is_file():
        supplied = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise SafetyError("Slab-alignment configuration must be a JSON object")
        config.update(supplied)
    if config["side"] not in ("high-z", "low-z"):
        raise SafetyError("Configuration side must be high-z or low-z")
    if not isinstance(config["references"], list) or not config["references"]:
        raise SafetyError("Configuration references must be a nonempty list")
    return config


def reference_for(name: str, config: dict[str, Any]) -> str | None:
    for rule in config["references"]:
        prefix = rule["prefix"]
        if name == prefix or name.startswith(prefix + "_"):
            return str(rule["reference"])
    return None


def plateau_status(plateau: Plateau, config: dict[str, Any]) -> str:
    if plateau.swing_eV >= config["swing_fail_eV"] or plateau.residual_std_eV >= config["std_fail_eV"]:
        return "FAILED_FLATNESS"
    if plateau.swing_eV >= config["swing_warn_eV"] or plateau.residual_std_eV >= config["std_warn_eV"]:
        return "SUSPECT_FLATNESS"
    return "OK"


def _plot_profile(
    calc_dir: Path,
    shifted_z: np.ndarray,
    potential: np.ndarray,
    profile: ProfileAnalysis,
    selected: Plateau,
    efermi: float | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError("matplotlib is required for slab alignment. Install interfaceforge[slab-align].") from exc
    figure, axes = plt.subplots(figsize=(8.0, 4.8))
    offset = efermi if efermi is not None else 0.0
    axes.plot(shifted_z, potential - offset, color="black", linewidth=1.4)
    axes.axvspan(
        selected.window_start_A,
        selected.window_end_A,
        color="#F59E0B",
        alpha=0.18,
        label=f"selected {selected.side} fit window",
    )
    axes.axvspan(
        profile.atom_low_A,
        profile.atom_high_A,
        color="0.7",
        alpha=0.18,
        label="atomic region",
    )
    if selected.correction_step_detected and selected.correction_step_A is not None:
        axes.axvline(
            selected.correction_step_A,
            color="#7C3AED",
            linestyle=":",
            linewidth=1.2,
            label="dipole-correction step (excluded)",
        )
    axes.set(
        xlabel="Shifted distance along c (Å)",
        ylabel=(
            r"Planar-averaged local potential, $\overline{V}_{\rm loc}(z)-E_F$ (eV)"
            if efermi is not None
            else r"Planar-averaged local potential, $\overline{V}_{\rm loc}(z)$ (eV)"
        ),
        xlim=(0, profile.c_length_A),
        title=calc_dir.name,
    )
    axes.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(calc_dir / "vacuum_profile.png", dpi=250)
    plt.close(figure)


def _plot_workfunction_profile(
    calc_dir: Path,
    shifted_z: np.ndarray,
    potential: np.ndarray,
    profile: ProfileAnalysis,
    selected: Plateau,
    efermi: float | None,
) -> None:
    """Write a simple work-function-style profile adapted from the user tool."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError("matplotlib is required for slab alignment. Install interfaceforge[slab-align].") from exc
    offset = efermi if efermi is not None else 0.0
    values = potential - offset
    plateau = selected.plateau_eV - offset
    figure, axes = plt.subplots(figsize=(8.0, 4.8))
    axes.plot(shifted_z, values, color="black", linewidth=1.2)
    axes.axhline(plateau, color="#B91C1C", linestyle="--", linewidth=1.0, label="selected vacuum")
    axes.axvspan(
        selected.window_start_A,
        selected.window_end_A,
        color="#F59E0B",
        alpha=0.18,
        label=f"{selected.side} fit window",
    )
    if selected.correction_step_detected and selected.correction_step_A is not None:
        axes.axvline(
            selected.correction_step_A,
            color="#7C3AED",
            linestyle=":",
            linewidth=1.2,
            label="dipole-correction step (excluded)",
        )
    axes.grid(color="gray", linestyle="-.", alpha=0.45)
    axes.minorticks_on()
    axes.set_xlim(0, profile.c_length_A)
    upper = float(np.max(values)) + 0.5
    axes.set_ylim(upper - 2.5, upper)
    axes.set_xlabel("Shifted distance along c (Å)")
    axes.set_ylabel(
        r"$\overline{V}_{\rm loc}(z)-E_F$ (eV)"
        if efermi is not None
        else r"$\overline{V}_{\rm loc}(z)$ (eV)"
    )
    axes.set_title(calc_dir.name)
    axes.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(calc_dir / "Workfunction.png", dpi=400)
    plt.close(figure)


def _remove_generated_marker(calc_dir: Path, name: str) -> None:
    marker = calc_dir / name
    if marker.is_file():
        marker.unlink()


def _write_audit_markers(
    calc_dir: Path,
    row: dict[str, Any],
    *,
    write_fix: bool,
) -> None:
    """Flag the calculation without modifying INCAR or submitting anything."""

    flatness = str(row.get("flatness_status", "FAILED_ANALYSIS"))
    for marker in (OK_MARKER, REVIEW_MARKER, AUDIT_FAILED_MARKER):
        if marker != (
            OK_MARKER
            if flatness == "OK"
            else AUDIT_FAILED_MARKER
            if flatness == "FAILED_ANALYSIS"
            else REVIEW_MARKER
        ):
            _remove_generated_marker(calc_dir, marker)

    if flatness == "OK":
        # A case can move from flagged to accepted after a corrected audit.
        # Remove the obsolete generated proposal so it cannot be mistaken for
        # a still-recommended INCAR change.
        _remove_generated_marker(calc_dir, "INCAR.dipole_fix")
        marker = calc_dir / OK_MARKER
        marker.write_text(
            "LOCPOT selected-side vacuum passed the InterfaceForge flatness audit.\n",
            encoding="utf-8",
        )
        row["audit_action"] = "NONE_FLAT_ENOUGH"
        row["relaunch_review_required"] = False
        row["review_flag_path"] = str(marker)
        return

    if flatness == "FAILED_ANALYSIS":
        marker = calc_dir / AUDIT_FAILED_MARKER
        marker.write_text(
            "InterfaceForge could not audit this LOCPOT. Review the reported error before relaunch.\n"
            f"error: {row.get('error', '')}\n",
            encoding="utf-8",
        )
        row["audit_action"] = "REVIEW_AUDIT_FAILURE"
        row["relaunch_review_required"] = True
        row["review_flag_path"] = str(marker)
        return

    fix_path = write_dipole_preview(calc_dir, float(row["suggested_DIPOL_z"])) if write_fix else None
    marker = calc_dir / REVIEW_MARKER
    marker.write_text(
        "LOCPOT vacuum is not flat enough for automatic acceptance.\n"
        f"flatness_status: {flatness}\n"
        f"selected_side: {row.get('selected_side', '')}\n"
        f"selected_swing_eV: {row.get('selected_swing_eV', '')}\n"
        f"selected_std_eV: {row.get('selected_std_eV', '')}\n"
        f"suggested_DIPOL_z: {row.get('suggested_DIPOL_z', '')}\n"
        f"compactness_R: {row.get('compactness_R', '')}\n"
        f"proposed_incar: {fix_path or 'disabled'}\n"
        "No calculation was submitted and INCAR was not modified.\n",
        encoding="utf-8",
    )
    row["audit_action"] = "REVIEW_PROPOSED_INCAR"
    row["relaunch_review_required"] = True
    row["dipole_fix_path"] = str(fix_path) if fix_path else ""
    row["review_flag_path"] = str(marker)


def _run_sumo(calc_dir: Path) -> str:
    executable = shutil.which("sumo-dosplot")
    if not executable:
        return "NOT_FOUND"
    with (calc_dir / "sumo_dosplot.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [executable],
            cwd=calc_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return "OK" if result.returncode == 0 else f"FAILED_{result.returncode}"


def _analyze_folder(
    calc_dir: Path,
    config: dict[str, Any],
    *,
    run_sumo: bool,
    write_fixes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row: dict[str, Any] = {
        "folder": calc_dir.name,
        "reference": reference_for(calc_dir.name, config) or "",
        "status": "FAILED_ANALYSIS",
        "error": "",
        "flatness_status": "FAILED_ANALYSIS",
        "flat_enough": False,
        "audit_action": "REVIEW_AUDIT_FAILURE",
        "relaunch_review_required": True,
        "dipole_fix_path": "",
        "review_flag_path": "",
        "band_edge_status": "NOT_ANALYZED",
        "selected_side": config["side"],
        "pdos_review_required": True,
        "sumo_status": "NOT_REQUESTED",
    }
    details: dict[str, Any] = {}
    try:
        structure, z_grid, potential = read_locpot(calc_dir / "LOCPOT")
        profile, shifted_z, shifted_potential = analyze_profile(
            structure,
            z_grid,
            potential,
            buffer_angstrom=config["buffer_angstrom"],
            minimum_window_angstrom=config["minimum_window_angstrom"],
            discontinuity_min_eV=config["discontinuity_min_eV"],
            discontinuity_factor=config["discontinuity_factor"],
            discontinuity_max_width_angstrom=config["discontinuity_max_width_angstrom"],
            discontinuity_margin_angstrom=config["discontinuity_margin_angstrom"],
        )
        selected = profile.high if config["side"] == "high-z" else profile.low
        selected_status = plateau_status(selected, config)
        center, compactness, missing_mass = ionic_center_fraction(structure)
        incar = parse_incar(calc_dir / "INCAR")
        row.update(
            {
                "status": selected_status,
                "flatness_status": selected_status,
                "flat_enough": selected_status == "OK",
                "vacuum_eV": selected.plateau_eV,
                "selected_slope_eV_per_A": selected.slope_eV_per_A,
                "selected_swing_eV": selected.swing_eV,
                "selected_std_eV": selected.residual_std_eV,
                "selected_correction_step_detected": selected.correction_step_detected,
                "selected_correction_step_A": selected.correction_step_A,
                "selected_correction_step_eV": selected.correction_step_eV,
                "selected_correction_step_width_A": selected.correction_step_width_A,
                "low_vacuum_eV": profile.low.plateau_eV,
                "high_vacuum_eV": profile.high.plateau_eV,
                "high_minus_low_vacuum_eV": profile.high.plateau_eV - profile.low.plateau_eV,
                "low_slope_eV_per_A": profile.low.slope_eV_per_A,
                "high_slope_eV_per_A": profile.high.slope_eV_per_A,
                "suggested_DIPOL_z": center,
                "compactness_R": compactness,
                "current_LDIPOL": incar["LDIPOL"],
                "current_IDIPOL": incar["IDIPOL"],
                "current_DIPOL": incar["DIPOL"],
            }
        )
        if missing_mass:
            row["error"] = "Missing masses: " + ",".join(missing_mass)
        try:
            efermi_outcar = efermi_from_outcar(calc_dir / "OUTCAR")
        except (OSError, SafetyError):
            efermi_outcar = None
        _plot_profile(calc_dir, shifted_z, shifted_potential, profile, selected, efermi_outcar)
        _plot_workfunction_profile(
            calc_dir,
            shifted_z,
            shifted_potential,
            profile,
            selected,
            efermi_outcar,
        )
        data = "\n".join(
            (
                f"{position:.8f} {value:.8f} {value - efermi_outcar:.8f}"
                if efermi_outcar is not None
                else f"{position:.8f} {value:.8f} nan"
            )
            for position, value in zip(shifted_z, shifted_potential, strict=True)
        )
        (calc_dir / "locpot.dat").write_text(
            "# shifted_z_A potential_eV potential_minus_EF_eV\n" + data + "\n",
            encoding="utf-8",
        )
        _write_audit_markers(calc_dir, row, write_fix=write_fixes)

        try:
            if efermi_outcar is None:
                efermi_outcar = efermi_from_outcar(calc_dir / "OUTCAR")
            efermi_xml, vbm, cbm, gap = band_edges_from_vasprun(calc_dir / "vasprun.xml")
            if abs(efermi_xml - efermi_outcar) > 1e-3:
                row["band_edge_status"] = "FAILED_EFERMI_MISMATCH"
                row["status"] = "FAILED_EFERMI_MISMATCH"
                row["error"] = f"OUTCAR/XML E-fermi differ by {efermi_xml - efermi_outcar:.6f} eV"
            else:
                row.update(
                    {
                        "band_edge_status": "OK",
                        "efermi_eV": efermi_xml,
                        "vacuum_minus_ef_eV": selected.plateau_eV - efermi_outcar,
                        "vbm_eV": vbm,
                        "cbm_eV": cbm,
                        "gap_eV": gap,
                        "vbm_vac_eV": vbm - selected.plateau_eV,
                        "cbm_vac_eV": cbm - selected.plateau_eV,
                    }
                )
        except (OSError, ValueError, ET.ParseError, SafetyError) as exc:
            row["band_edge_status"] = "FAILED_BAND_EDGES"
            row["status"] = "FAILED_BAND_EDGES"
            note = str(exc).replace("\t", " ").replace("\n", " ")
            row["error"] = f"{row['error']}; {note}".strip("; ")
        if run_sumo:
            row["sumo_status"] = _run_sumo(calc_dir)
        details = {
            "folder": calc_dir.name,
            "profile": asdict(profile),
            "selected_side": config["side"],
            "suggested_DIPOL_z": center,
            "compactness_R": compactness,
        }
    except (OSError, ValueError, ET.ParseError, SafetyError, DependencyError) as exc:
        row["status"] = "FAILED_ANALYSIS"
        row["flatness_status"] = "FAILED_ANALYSIS"
        row["error"] = str(exc).replace("\t", " ").replace("\n", " ")
        _write_audit_markers(calc_dir, row, write_fix=False)
    return row, details


def add_alignment_deltas(rows: list[dict[str, Any]]) -> None:
    by_name = {row["folder"]: row for row in rows}
    for row in rows:
        row["delta_vbm_eV"] = ""
        row["delta_cbm_eV"] = ""
        reference = by_name.get(row.get("reference", ""))
        if row.get("status") not in ("OK", "SUSPECT_FLATNESS") or not reference:
            continue
        if reference.get("status") not in ("OK", "SUSPECT_FLATNESS"):
            continue
        row["delta_vbm_eV"] = row["vbm_vac_eV"] - reference["vbm_vac_eV"]
        row["delta_cbm_eV"] = row["cbm_vac_eV"] - reference["cbm_vac_eV"]


def _format_field(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _write_outputs(root: Path, rows: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, str]:
    tsv_path = root / "band_edge_alignment.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_field(row.get(field, "")) for field in OUTPUT_FIELDS})
    json_path = root / "band_edge_alignment.json"
    json_path.write_text(
        json.dumps({"rows": rows, "details": details}, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path = root / "band_edge_alignment.txt"
    lines = [
        "Vacuum-aligned slab band edges",
        "===============================",
        "Positive delta means movement upward, toward vacuum.",
        "Global VASP edges are reported; inspect SUMO PDOS before assigning a perovskite-derived CBM.",
        "",
        f"{'folder':40s} {'status':21s} {'CBMvac':>10s} {'dCBM':>10s} {'VBMvac':>10s} {'dVBM':>10s} {'swing':>9s}",
    ]

    def number(row: dict[str, Any], key: str) -> str:
        value = row.get(key, "")
        return f"{value:10.4f}" if isinstance(value, float) else f"{'--':>10s}"

    for row in rows:
        lines.append(
            f"{row['folder'][:40]:40s} {row['status'][:21]:21s} "
            f"{number(row, 'cbm_vac_eV')} {number(row, 'delta_cbm_eV')} "
            f"{number(row, 'vbm_vac_eV')} {number(row, 'delta_vbm_eV')} "
            f"{number(row, 'selected_swing_eV')}"
        )
        if row.get("error"):
            lines.append(f"  note: {row['error']}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_tsv = root / "dipole_flatness_audit.tsv"
    with audit_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_field(row.get(field, "")) for field in AUDIT_FIELDS})
    audit_text = root / "dipole_flatness_audit.txt"
    audit_lines = [
        "LOCPOT vacuum-flatness audit",
        "============================",
        "OK calculations need no dipole improvement. REVIEW rows were not relaunched; inspect the proposed INCAR.",
        "",
        f"{'folder':40s} {'flatness':20s} {'swing/eV':>10s} {'std/eV':>10s} {'action':>24s}",
    ]
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("relaunch_review_required"):
            review_rows.append(row)
        audit_lines.append(
            f"{row['folder'][:40]:40s} {str(row.get('flatness_status', ''))[:20]:20s} "
            f"{_format_field(row.get('selected_swing_eV', '')):>10.10s} "
            f"{_format_field(row.get('selected_std_eV', '')):>10.10s} "
            f"{str(row.get('audit_action', '')):>24s}"
        )
    audit_text.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    review_queue = root / "relaunch_review_queue.txt"
    queue_lines = [
        "InterfaceForge relaunch review queue",
        "====================================",
        "This is a review queue only. No INCAR was overwritten and no VASP job was submitted.",
        "",
    ]
    if not review_rows:
        queue_lines.append("All analyzed LOCPOT selected-side plateaus are flat enough; no relaunch review is needed.")
    for row in review_rows:
        queue_lines.extend(
            [
                row["folder"],
                f"  flatness: {row.get('flatness_status', '')}",
                f"  action: {row.get('audit_action', '')}",
                f"  flag: {row.get('review_flag_path', '')}",
                f"  proposed INCAR: {row.get('dipole_fix_path', '') or 'not available'}",
            ]
        )
        if row.get("dipole_fix_path"):
            queue_lines.append(
                f"  inspect: diff -u {row['folder']}/INCAR {row['folder']}/INCAR.dipole_fix"
            )
        queue_lines.append("")
    review_queue.write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    return {
        "tsv": str(tsv_path),
        "json": str(json_path),
        "text": str(text_path),
        "flatness_tsv": str(audit_tsv),
        "flatness_text": str(audit_text),
        "review_queue": str(review_queue),
    }


def analyze_slab_alignment(
    root: str | Path = ".",
    *,
    config: str | Path = "slab_alignment.json",
    run_sumo: bool = False,
    write_dipole_fixes: bool = True,
    only: str | None = None,
) -> dict[str, Any]:
    """Analyze all configured immediate child calculations and align band edges."""

    root_path = Path(root).expanduser().resolve()
    config_path = Path(config).expanduser()
    if not config_path.is_absolute():
        config_path = root_path / config_path
    settings = load_alignment_config(config_path)
    calculation_dirs = sorted(
        path
        for path in root_path.iterdir()
        if path.is_dir()
        and (path / "LOCPOT").is_file()
        and reference_for(path.name, settings)
        and (only is None or path.name == only)
    )
    if not calculation_dirs:
        raise SafetyError("No matching immediate subfolder contains LOCPOT")
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for calculation_dir in calculation_dirs:
        row, detail = _analyze_folder(
            calculation_dir,
            settings,
            run_sumo=run_sumo,
            write_fixes=write_dipole_fixes,
        )
        rows.append(row)
        if detail:
            details.append(detail)
    add_alignment_deltas(rows)
    outputs = _write_outputs(root_path, rows, details)
    failures = sum(row["status"].startswith("FAILED") for row in rows)
    suspects = sum(row["status"] == "SUSPECT_FLATNESS" for row in rows)
    flat_enough = sum(row.get("flatness_status") == "OK" for row in rows)
    review_required = sum(bool(row.get("relaunch_review_required")) for row in rows)
    return {
        "root": str(root_path),
        "config": str(config_path),
        "count": len(rows),
        "failures": failures,
        "suspects": suspects,
        "flat_enough": flat_enough,
        "review_required": review_required,
        "status": "FAILED" if failures else ("SUSPECT" if suspects else "OK"),
        "outputs": outputs,
        "rows": rows,
    }
