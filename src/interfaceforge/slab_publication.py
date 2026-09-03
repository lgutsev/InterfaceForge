"""Publication figures for vacuum-aligned VASP slab calculations."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError
from .slab_alignment import (
    Plateau,
    ProfileAnalysis,
    Structure,
    analyze_profile,
    band_edges_from_vasprun,
    efermi_from_outcar,
    largest_periodic_gap,
    plateau_status,
    read_locpot,
)


@dataclass
class PublicationCase:
    name: str
    path: Path
    structure: Structure
    profile: ProfileAnalysis
    selected: Plateau
    shifted_z: np.ndarray
    potential_minus_ef: np.ndarray
    efermi_eV: float
    vacuum_eV: float
    vbm_vac_eV: float
    cbm_vac_eV: float
    flatness_status: str


def load_publication_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the deliberately small publication-case manifest."""

    defaults: dict[str, Any] = {
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
        "pairs": [],
        "framework_elements": ["Pb", "I"],
        "passivant_elements": ["C", "N", "O", "Br", "H"],
        "passivant_label": "BPDCA",
        "energy_window_eV": [-7.0, -2.0],
        "sumo_gaussian_eV": 0.05,
        "atom_match_tolerance_angstrom": 1.5,
        "vacuum_context_angstrom": 2.0,
        "dpi": 600,
        "allow_suspect": False,
    }
    input_path = Path(path)
    if not input_path.is_file():
        raise SafetyError(f"Publication configuration not found: {input_path}")
    supplied = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise SafetyError("Publication configuration must be a JSON object")
    config = defaults | supplied
    if config["side"] not in ("high-z", "low-z"):
        raise SafetyError("Publication side must be high-z or low-z")
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise SafetyError("Publication configuration requires a nonempty pairs list")
    for pair in pairs:
        if not isinstance(pair, dict) or not all(key in pair for key in ("label", "reference", "passivated")):
            raise SafetyError("Every publication pair needs label, reference, and passivated")
    window = config.get("energy_window_eV")
    if not isinstance(window, list) or len(window) != 2 or float(window[0]) >= float(window[1]):
        raise SafetyError("energy_window_eV must be [minimum, maximum]")
    if float(config["vacuum_context_angstrom"]) < 0:
        raise SafetyError("vacuum_context_angstrom must be nonnegative")
    return config


def _load_case(root: Path, name: str, config: dict[str, Any]) -> PublicationCase:
    calc_dir = root / name
    required = [calc_dir / item for item in ("LOCPOT", "OUTCAR", "vasprun.xml")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise SafetyError(f"{name}: missing {', '.join(missing)}")
    structure, z_grid, potential = read_locpot(calc_dir / "LOCPOT")
    profile, shifted_z, shifted_potential = analyze_profile(
        structure,
        z_grid,
        potential,
        buffer_angstrom=float(config["buffer_angstrom"]),
        minimum_window_angstrom=float(config["minimum_window_angstrom"]),
        discontinuity_min_eV=float(config["discontinuity_min_eV"]),
        discontinuity_factor=float(config["discontinuity_factor"]),
        discontinuity_max_width_angstrom=float(config["discontinuity_max_width_angstrom"]),
        discontinuity_margin_angstrom=float(config["discontinuity_margin_angstrom"]),
    )
    selected = profile.high if config["side"] == "high-z" else profile.low
    status = plateau_status(selected, config)
    accepted = status == "OK" or (status == "SUSPECT_FLATNESS" and config["allow_suspect"])
    if not accepted:
        raise SafetyError(
            f"{name}: selected plateau is {status} "
            f"(swing={selected.swing_eV:.4f} eV, std={selected.residual_std_eV:.4f} eV)"
        )
    efermi_outcar = efermi_from_outcar(calc_dir / "OUTCAR")
    efermi_xml, vbm, cbm, _gap = band_edges_from_vasprun(calc_dir / "vasprun.xml")
    if abs(efermi_xml - efermi_outcar) > 1e-3:
        raise SafetyError(
            f"{name}: OUTCAR/XML E-fermi differ by {efermi_xml - efermi_outcar:.6f} eV"
        )
    return PublicationCase(
        name=name,
        path=calc_dir,
        structure=structure,
        profile=profile,
        selected=selected,
        shifted_z=shifted_z,
        potential_minus_ef=shifted_potential - efermi_outcar,
        efermi_eV=efermi_outcar,
        vacuum_eV=selected.plateau_eV,
        vbm_vac_eV=vbm - selected.plateau_eV,
        cbm_vac_eV=cbm - selected.plateau_eV,
        flatness_status=status,
    )


def _atom_matching_geometry(
    reference: Structure,
    passivated: Structure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build safe coordinates for slabs with independently padded vacuum.

    Adsorbate builders commonly preserve the in-plane surface lattice while
    choosing a different c length to retain a target amount of vacuum. That
    does not invalidate species-local atom matching. In-plane changes or a
    rotated surface normal remain unsafe and are rejected.
    """

    if not np.allclose(reference.cell[:2], passivated.cell[:2], atol=0.05, rtol=1e-3):
        maximum_delta = float(np.max(np.abs(reference.cell[:2] - passivated.cell[:2])))
        raise SafetyError(
            "Reference and passivated in-plane cells differ "
            f"(maximum component difference {maximum_delta:.4f} A); ligand atom matching is unsafe"
        )

    reference_c = reference.cell[2]
    passivated_c = passivated.cell[2]
    reference_length = float(np.linalg.norm(reference_c))
    passivated_length = float(np.linalg.norm(passivated_c))
    direction_cosine = float(
        np.dot(reference_c, passivated_c) / (reference_length * passivated_length)
    )
    if direction_cosine < 0.9999:
        raise SafetyError(
            "Reference and passivated surface-normal directions differ; "
            "ligand atom matching is unsafe"
        )
    reference_counts = dict(zip(reference.species, reference.counts, strict=True))
    passivated_counts = dict(zip(passivated.species, passivated.counts, strict=True))
    anchor_symbols = {
        symbol
        for symbol, count in reference_counts.items()
        if count > 0 and passivated_counts.get(symbol) == count
    }
    if not anchor_symbols:
        raise SafetyError(
            "Reference and passivated structures have no unchanged species for slab alignment"
        )

    def centered_z(structure: Structure) -> np.ndarray:
        gap_start, _gap_end, gap_width = largest_periodic_gap(
            structure.z_angstrom, structure.c_length
        )
        vacuum_center = (gap_start + 0.5 * gap_width) % structure.c_length
        unwrapped = np.mod(structure.z_angstrom - vacuum_center, structure.c_length)
        anchors = np.asarray(
            [symbol in anchor_symbols for symbol in structure.elements], dtype=bool
        )
        return unwrapped - float(np.median(unwrapped[anchors]))

    average_in_plane = 0.5 * (reference.cell[:2] + passivated.cell[:2])
    c_direction = reference_c / reference_length
    return average_in_plane, c_direction, centered_z(reference), centered_z(passivated)


def _matching_distance(
    first: np.ndarray,
    first_z: float,
    second: np.ndarray,
    second_z: float,
    in_plane_cell: np.ndarray,
    c_direction: np.ndarray,
) -> float:
    delta_xy = np.asarray(first[:2]) - np.asarray(second[:2])
    delta_xy -= np.round(delta_xy)
    displacement = delta_xy @ in_plane_cell + (first_z - second_z) * c_direction
    return float(np.linalg.norm(displacement))


def match_excess_atoms(
    reference: Structure,
    passivated: Structure,
    *,
    tolerance_angstrom: float = 1.5,
) -> dict[str, list[int]]:
    """Return 1-based species-local indices added to the passivated slab.

    Matching by geometry, rather than assuming appended POSCAR ordering, keeps
    methylammonium C/N/H out of the BPDCA projection.
    """

    in_plane_cell, c_direction, reference_z, passivated_z = _atom_matching_geometry(
        reference, passivated
    )
    ref_by_symbol: dict[str, list[tuple[np.ndarray, float]]] = {}
    pass_by_symbol: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    ref_local: dict[str, int] = {}
    pass_local: dict[str, int] = {}
    for symbol, coordinate, z_value in zip(
        reference.elements, reference.fractional, reference_z, strict=True
    ):
        ref_local[symbol] = ref_local.get(symbol, 0) + 1
        ref_by_symbol.setdefault(symbol, []).append((coordinate, float(z_value)))
    for symbol, coordinate, z_value in zip(
        passivated.elements, passivated.fractional, passivated_z, strict=True
    ):
        pass_local[symbol] = pass_local.get(symbol, 0) + 1
        pass_by_symbol.setdefault(symbol, []).append(
            (pass_local[symbol], coordinate, float(z_value))
        )

    excess: dict[str, list[int]] = {}
    for symbol, candidates in pass_by_symbol.items():
        unmatched = list(candidates)
        for ref_coordinate, ref_z in ref_by_symbol.get(symbol, []):
            if not unmatched:
                raise SafetyError(f"Passivated structure contains fewer {symbol} atoms than its reference")
            distances = [
                _matching_distance(
                    ref_coordinate,
                    ref_z,
                    coordinate,
                    z_value,
                    in_plane_cell,
                    c_direction,
                )
                for _index, coordinate, z_value in unmatched
            ]
            best = int(np.argmin(distances))
            if distances[best] > tolerance_angstrom:
                raise SafetyError(
                    f"Could not match reference {symbol} atom within {tolerance_angstrom:.2f} A; "
                    "check that the structures share an origin"
                )
            unmatched.pop(best)
        if unmatched:
            excess[symbol] = [index for index, _coordinate, _z_value in unmatched]
    return excess


def _sumo_atom_selection(
    framework_elements: list[str],
    passivant_elements: list[str],
    excess: dict[str, list[int]],
) -> tuple[str, str]:
    ligand_symbols = [symbol for symbol in passivant_elements if excess.get(symbol)]
    elements = list(dict.fromkeys(framework_elements + ligand_symbols))
    atom_parts = list(framework_elements)
    for symbol in passivant_elements:
        indices = excess.get(symbol, [])
        if indices:
            atom_parts.append(symbol + "." + ".".join(str(index) for index in indices))
    return ",".join(elements), ",".join(atom_parts)


def _run_sumo(
    case: PublicationCase,
    *,
    framework_elements: list[str],
    passivant_elements: list[str],
    excess: dict[str, list[int]],
    gaussian_eV: float,
) -> Path:
    executable = shutil.which("sumo-dosplot")
    if not executable:
        raise DependencyError("sumo-dosplot was not found on PATH")
    data_dir = case.path / "publication_dos_data"
    data_dir.mkdir(exist_ok=True)
    elements, atoms = _sumo_atom_selection(framework_elements, passivant_elements, excess)
    command = [
        executable,
        "--no-shift",
        "--directory",
        str(data_dir),
        "--format",
        "pdf",
        "--elements",
        elements,
        "--atoms",
        atoms,
        "--gaussian",
        str(gaussian_eV),
    ]
    with (case.path / "publication_sumo_dosplot.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=case.path,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise SafetyError(
            f"sumo-dosplot failed for {case.name} with exit code {result.returncode}; "
            "see publication_sumo_dosplot.log"
        )
    return data_dir


def read_sumo_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a SUMO *_dos.dat file and combine all projected DOS columns."""

    input_path = Path(path)
    if not input_path.is_file():
        raise SafetyError(f"SUMO data file not found: {input_path}")
    values = np.loadtxt(input_path, comments="#", ndmin=2)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 2:
        raise SafetyError(f"SUMO data file has insufficient data: {input_path}")
    order = np.argsort(values[:, 0])
    energy = values[order, 0]
    density = np.sum(np.abs(values[order, 1:]), axis=1)
    return energy, density


def _find_sumo_file(data_dir: Path, stem: str) -> Path:
    direct = data_dir / f"{stem}_dos.dat"
    if direct.is_file():
        return direct
    matches = sorted(data_dir.glob(f"*{stem}_dos.dat"))
    if len(matches) == 1:
        return matches[0]
    raise SafetyError(f"Could not uniquely locate {stem}_dos.dat in {data_dir}")


def _interpolate_density(
    target_energy: np.ndarray,
    source_energy: np.ndarray,
    source_density: np.ndarray,
) -> np.ndarray:
    return np.interp(target_energy, source_energy, source_density, left=0.0, right=0.0)


def _load_pdos(
    case: PublicationCase,
    config: dict[str, Any],
    excess: dict[str, list[int]],
    *,
    run_sumo: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    framework = [str(item) for item in config["framework_elements"]]
    passivant = [str(item) for item in config["passivant_elements"]]
    data_dir = case.path / "publication_dos_data"
    if run_sumo:
        data_dir = _run_sumo(
            case,
            framework_elements=framework,
            passivant_elements=passivant,
            excess=excess,
            gaussian_eV=float(config["sumo_gaussian_eV"]),
        )
    if not data_dir.is_dir():
        raise SafetyError(
            f"{case.name}: {data_dir.name} is missing; rerun with --run-sumo on a compute node"
        )
    total_energy, total_density = read_sumo_curve(_find_sumo_file(data_dir, "total"))
    result = {"Total": (total_energy - case.vacuum_eV, total_density)}
    for symbol in framework:
        energy, density = read_sumo_curve(_find_sumo_file(data_dir, symbol))
        result[symbol] = (energy - case.vacuum_eV, density)
    ligand_density = np.zeros_like(total_density)
    found_ligand = False
    for symbol in passivant:
        if not excess.get(symbol):
            continue
        energy, density = read_sumo_curve(_find_sumo_file(data_dir, symbol))
        ligand_density += _interpolate_density(total_energy, energy, density)
        found_ligand = True
    if found_ligand:
        result[str(config["passivant_label"])] = (
            total_energy - case.vacuum_eV,
            ligand_density,
        )
    return result


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "matplotlib is required for publication plots. Install interfaceforge[slab-align]."
        ) from exc
    return plt


def _selected_side_plot_window(case: PublicationCase, context_angstrom: float) -> tuple[float, float]:
    """Return a publication crop containing only the selected vacuum side.

    Context is added toward the adjacent surface, never toward the periodic
    seam or a detected dipole-correction reset.  The fitted plateau boundary
    therefore remains the outer edge of the plotted data.
    """

    if context_angstrom < 0:
        raise SafetyError("vacuum_context_angstrom must be nonnegative")
    if case.selected.side == "high-z":
        start = max(0.0, case.selected.window_start_A - context_angstrom)
        end = case.selected.window_end_A
    else:
        start = case.selected.window_start_A
        end = min(case.profile.c_length_A, case.selected.window_end_A + context_angstrom)
    if end <= start:
        raise SafetyError(f"{case.name}: selected-side publication crop is empty")
    return start, end


def _plot_vacuum(
    pair_cases: list[tuple[str, PublicationCase, PublicationCase]],
    config: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    plt = _matplotlib()
    figure, axes = plt.subplots(len(pair_cases), 2, figsize=(7.2, 2.75 * len(pair_cases)), squeeze=False)
    context_angstrom = float(config["vacuum_context_angstrom"])
    panel_index = 0
    for row, (label, reference, passivated) in enumerate(pair_cases):
        for column, (kind, case) in enumerate((("Pristine", reference), ("BPDCA", passivated))):
            axis = axes[row, column]
            crop_start, crop_end = _selected_side_plot_window(case, context_angstrom)
            crop = (case.shifted_z >= crop_start) & (case.shifted_z <= crop_end)
            if np.count_nonzero(crop) < 5:
                raise SafetyError(
                    f"{case.name}: selected-side publication crop contains fewer than five points"
                )
            axis.plot(
                case.shifted_z[crop],
                case.potential_minus_ef[crop],
                color="black",
                linewidth=1.15,
            )
            axis.axvspan(
                case.selected.window_start_A,
                case.selected.window_end_A,
                color="#E69F00",
                alpha=0.20,
            )
            plateau = case.selected.plateau_eV - case.efermi_eV
            axis.hlines(
                plateau,
                case.selected.window_start_A,
                case.selected.window_end_A,
                color="#C62828",
                linestyle="--",
                linewidth=1.0,
            )
            axis.text(
                0.03,
                0.94,
                rf"$\Phi_{{\rm {case.selected.side}}}={plateau:.2f}$ eV"
                + "\n"
                + rf"swing $={case.selected.swing_eV:.3f}$ eV",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            axis.set_title(f"{label}: {kind}", fontsize=10)
            axis.text(
                0.01,
                1.03,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
            panel_index += 1
            axis.set_xlim(crop_start, crop_end)
            axis.set_xlabel(r"Shifted distance along $c$ ($\AA$)")
            if column == 0:
                axis.set_ylabel(r"$\overline{V}_{\rm loc}(z)-E_F$ (eV)")
            axis.tick_params(direction="in", top=True, right=True)
        lower = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
        upper = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        axes[row, 0].set_ylim(lower, upper)
        axes[row, 1].set_ylim(lower, upper)
    figure.tight_layout()
    outputs: dict[str, str] = {}
    for extension in ("pdf", "png", "svg"):
        path = output_dir / f"vacuum_validation.{extension}"
        figure.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        outputs[extension] = str(path)
    plt.close(figure)
    return outputs


def _plot_pdos_axis(
    axis: Any,
    case: PublicationCase,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    window: tuple[float, float],
    title: str,
    scale: float,
) -> None:
    colors = {"Total": "#111111", "Pb": "#0072B2", "I": "#CC79A7", "BPDCA": "#E69F00"}
    if scale <= 0:
        raise SafetyError(f"{case.name}: total DOS is zero in the requested plot range")
    for label in ("Total", "Pb", "I", "BPDCA"):
        if label not in curves:
            continue
        energy, density = curves[label]
        axis.plot(energy, density / scale, color=colors[label], linewidth=1.15, label=label)
    axis.axvline(
        case.vbm_vac_eV,
        color="#009E73",
        linestyle="--",
        linewidth=0.9,
        label="VBM",
    )
    axis.axvline(
        case.cbm_vac_eV,
        color="#D55E00",
        linestyle="--",
        linewidth=0.9,
        label="CBM",
    )
    axis.set_xlim(*window)
    axis.set_ylim(bottom=0)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(r"$E-V_{\rm vac}$ (eV)")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False, fontsize=7, ncol=3)


def _plot_level_axis(
    axis: Any,
    reference: PublicationCase,
    passivated: PublicationCase,
    window: tuple[float, float],
) -> None:
    x_ref, x_pass = 0.25, 0.75
    for energy, color in (
        (reference.vbm_vac_eV, "#009E73"),
        (reference.cbm_vac_eV, "#D55E00"),
    ):
        axis.hlines(energy, x_ref - 0.14, x_ref + 0.14, color=color, linewidth=2.0)
    for energy, color in (
        (passivated.vbm_vac_eV, "#009E73"),
        (passivated.cbm_vac_eV, "#D55E00"),
    ):
        axis.hlines(energy, x_pass - 0.14, x_pass + 0.14, color=color, linewidth=2.0)
    axis.plot(
        [x_ref + 0.14, x_pass - 0.14],
        [reference.vbm_vac_eV, passivated.vbm_vac_eV],
        color="#009E73",
        linewidth=0.8,
        alpha=0.7,
    )
    axis.plot(
        [x_ref + 0.14, x_pass - 0.14],
        [reference.cbm_vac_eV, passivated.cbm_vac_eV],
        color="#D55E00",
        linewidth=0.8,
        alpha=0.7,
    )
    delta_vbm = passivated.vbm_vac_eV - reference.vbm_vac_eV
    delta_cbm = passivated.cbm_vac_eV - reference.cbm_vac_eV
    axis.text(
        0.50,
        0.98,
        rf"$\Delta E_{{\rm VBM}}={delta_vbm:+.2f}$ eV" + "\n" + rf"$\Delta E_{{\rm CBM}}={delta_cbm:+.2f}$ eV",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(*window)
    axis.set_xticks([x_ref, x_pass], ["Pristine", "BPDCA"])
    axis.set_ylabel(r"$E-V_{\rm vac}$ (eV)")
    axis.set_title("Vacuum-aligned edges", fontsize=10)
    axis.tick_params(direction="in", right=True)


def _plot_electronic(
    pair_cases: list[tuple[str, PublicationCase, PublicationCase]],
    pdos: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    config: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    plt = _matplotlib()
    figure, axes = plt.subplots(len(pair_cases), 3, figsize=(10.5, 3.0 * len(pair_cases)), squeeze=False)
    window = tuple(float(value) for value in config["energy_window_eV"])
    panel_index = 0
    for row, (label, reference, passivated) in enumerate(pair_cases):
        pair_scale = 0.0
        for case in (reference, passivated):
            energy, density = pdos[case.name]["Total"]
            in_window = (energy >= window[0]) & (energy <= window[1])
            candidate = float(np.max(density[in_window])) if np.any(in_window) else float(np.max(density))
            pair_scale = max(pair_scale, candidate)
        _plot_pdos_axis(
            axes[row, 0],
            reference,
            pdos[reference.name],
            window,
            f"{label}: pristine",
            pair_scale,
        )
        _plot_pdos_axis(
            axes[row, 1],
            passivated,
            pdos[passivated.name],
            window,
            f"{label}: BPDCA",
            pair_scale,
        )
        _plot_level_axis(axes[row, 2], reference, passivated, window)
        axes[row, 0].set_ylabel("Normalized DOS")
        shared_ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        axes[row, 0].set_ylim(0, shared_ymax)
        axes[row, 1].set_ylim(0, shared_ymax)
        for column in range(3):
            axes[row, column].text(
                0.01,
                1.03,
                f"({chr(ord('a') + panel_index)})",
                transform=axes[row, column].transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
            panel_index += 1
    figure.tight_layout()
    outputs: dict[str, str] = {}
    for extension in ("pdf", "png", "svg"):
        path = output_dir / f"electronic_alignment.{extension}"
        figure.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        outputs[extension] = str(path)
    plt.close(figure)
    return outputs


def plot_slab_publication(
    root: str | Path = ".",
    *,
    config: str | Path = "slab_publication.json",
    output_dir: str | Path = "publication_figures",
    run_sumo: bool = False,
) -> dict[str, Any]:
    """Create publication-ready vacuum and PDOS/alignment figures."""

    root_path = Path(root).expanduser().resolve()
    config_path = Path(config).expanduser()
    if not config_path.is_absolute():
        config_path = root_path / config_path
    settings = load_publication_config(config_path)
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root_path / destination
    destination.mkdir(parents=True, exist_ok=True)

    pair_cases: list[tuple[str, PublicationCase, PublicationCase]] = []
    pdos: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    atom_selections: dict[str, dict[str, list[int]]] = {}
    for pair in settings["pairs"]:
        reference = _load_case(root_path, str(pair["reference"]), settings)
        passivated = _load_case(root_path, str(pair["passivated"]), settings)
        excess = match_excess_atoms(
            reference.structure,
            passivated.structure,
            tolerance_angstrom=float(settings["atom_match_tolerance_angstrom"]),
        )
        selected_excess = {
            symbol: indices
            for symbol, indices in excess.items()
            if symbol in settings["passivant_elements"]
        }
        unexpected = {
            symbol: indices
            for symbol, indices in excess.items()
            if symbol not in settings["passivant_elements"]
        }
        if unexpected:
            raise SafetyError(
                f"{passivated.name}: unexpected added atoms outside passivant_elements: {unexpected}"
            )
        atom_selections[reference.name] = {}
        atom_selections[passivated.name] = selected_excess
        pdos[reference.name] = _load_pdos(reference, settings, {}, run_sumo=run_sumo)
        pdos[passivated.name] = _load_pdos(
            passivated,
            settings,
            selected_excess,
            run_sumo=run_sumo,
        )
        pair_cases.append((str(pair["label"]), reference, passivated))

    dpi = int(settings["dpi"])
    vacuum_outputs = _plot_vacuum(pair_cases, settings, destination, dpi)
    electronic_outputs = _plot_electronic(pair_cases, pdos, settings, destination, dpi)
    rows: list[dict[str, Any]] = []
    for label, reference, passivated in pair_cases:
        rows.append(
            {
                "termination": label,
                "reference": reference.name,
                "passivated": passivated.name,
                "side": settings["side"],
                "reference_work_function_eV": reference.selected.plateau_eV - reference.efermi_eV,
                "passivated_work_function_eV": passivated.selected.plateau_eV - passivated.efermi_eV,
                "reference_vbm_vac_eV": reference.vbm_vac_eV,
                "passivated_vbm_vac_eV": passivated.vbm_vac_eV,
                "delta_vbm_eV": passivated.vbm_vac_eV - reference.vbm_vac_eV,
                "reference_cbm_vac_eV": reference.cbm_vac_eV,
                "passivated_cbm_vac_eV": passivated.cbm_vac_eV,
                "delta_cbm_eV": passivated.cbm_vac_eV - reference.cbm_vac_eV,
                "reference_swing_eV": reference.selected.swing_eV,
                "passivated_swing_eV": passivated.selected.swing_eV,
                "pdos_review_required": True,
            }
        )
    tsv_path = destination / "publication_band_edges.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = destination / "publication_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": settings,
                "rows": rows,
                "passivant_species_local_indices": atom_selections,
                "vacuum_figures": vacuum_outputs,
                "vacuum_figure_scope": {
                    "mode": "selected-side-only",
                    "side": settings["side"],
                    "surface_context_angstrom": settings["vacuum_context_angstrom"],
                    "note": (
                        "Each panel ends at the selected plateau boundary; the opposite-side "
                        "vacuum and dipole-correction reset are excluded."
                    ),
                },
                "electronic_figures": electronic_outputs,
                "interpretation_guard": (
                    "Global VASP VBM/CBM values are shown. Inspect the BPDCA PDOS before assigning "
                    "an apparent edge displacement to the perovskite-derived band edge."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "OK",
        "pairs": len(pair_cases),
        "output_dir": str(destination),
        "vacuum_figures": vacuum_outputs,
        "electronic_figures": electronic_outputs,
        "band_edges": str(tsv_path),
        "manifest": str(manifest_path),
    }
