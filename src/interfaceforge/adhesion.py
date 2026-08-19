"""Prepare a work-of-adhesion calculation tree from a reference VASP interface.

Given a finished (or in-progress) VASP-MLFF or DFT interface run, this module
prepares a sibling directory tree with everything needed to later compute the
work of adhesion: relaxed isolated-slab inputs for each fragment, and a rigid
separation curve (static single points at increasing rigid separation). The
reference directory is the zero-separation point and is never modified; this
module never launches VASP.

Ported from a standalone script (``split_vasp_interface.py``) that predates
its InterfaceForge integration; the geometry/INCAR logic is intentionally
close to the original.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SafetyError

EV_A2_TO_J_M2 = 16.02176634
METHODS = ("mlff", "dft")
Vec = tuple[float, float, float]
Lattice = tuple[Vec, Vec, Vec]


@dataclass
class Atom:
    symbol: str
    cart: Vec
    flags: tuple[str, ...]
    original_index: int


@dataclass
class Structure:
    title: str
    lattice: Lattice
    species: tuple[str, ...]
    atoms: list[Atom]
    selective: bool


def _vadd(a: Vec, b: Vec) -> Vec:
    return tuple(a[i] + b[i] for i in range(3))  # type: ignore[return-value]


def _vscale(v: Vec, factor: float) -> Vec:
    return tuple(factor * x for x in v)  # type: ignore[return-value]


def _dot(a: Vec, b: Vec) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec) -> float:
    return math.sqrt(_dot(v, v))


def _determinant(m: Lattice) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _frac_to_cart(frac: Vec, lattice: Lattice) -> Vec:
    return tuple(sum(frac[j] * lattice[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _cart_to_frac(cart: Vec, lattice: Lattice) -> Vec:
    # Solve lattice^T f = r by Cramer's rule.
    a, b, c = lattice
    m: Lattice = ((a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2]))
    d = _determinant(m)
    if abs(d) < 1e-14:
        raise ValueError("Singular lattice")
    x, y, z = cart
    return (
        (x * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
         - m[0][1] * (y * m[2][2] - m[1][2] * z)
         + m[0][2] * (y * m[2][1] - m[1][1] * z)) / d,
        (m[0][0] * (y * m[2][2] - m[1][2] * z)
         - x * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
         + m[0][2] * (m[1][0] * z - y * m[2][0])) / d,
        (m[0][0] * (m[1][1] * z - y * m[2][1])
         - m[0][1] * (m[1][0] * z - y * m[2][0])
         + x * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / d,
    )


def _read_vasp(path: Path) -> Structure:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"Invalid POSCAR/CONTCAR: {path}")
    factors = lines[1].split()
    if len(factors) != 1 or float(factors[0]) <= 0:
        raise ValueError("Only a single positive POSCAR scale factor is supported")
    factor = float(factors[0])
    lattice: Lattice = tuple(
        tuple(factor * float(x) for x in lines[i].split()[:3]) for i in range(2, 5)
    )  # type: ignore[assignment]
    species_fields = lines[5].split()
    try:
        [int(x) for x in species_fields]
    except ValueError:
        species = tuple(species_fields)
    else:
        raise ValueError("VASP 4 POSCARs without symbols are unsupported")
    counts = tuple(int(x) for x in lines[6].split())
    if len(species) != len(counts):
        raise ValueError("Species and count lines differ in length")
    cursor = 7
    selective = lines[cursor].strip().lower().startswith("s")
    if selective:
        cursor += 1
    mode = lines[cursor].strip().lower()
    if not mode or mode[0] not in {"d", "c", "k"}:
        raise ValueError("Coordinate mode must be Direct or Cartesian")
    direct = mode[0] == "d"
    cursor += 1
    atoms: list[Atom] = []
    index = 0
    for symbol, count in zip(species, counts, strict=True):
        for _ in range(count):
            fields = lines[cursor].split()
            xyz: Vec = tuple(float(x) for x in fields[:3])  # type: ignore[assignment]
            cart = _frac_to_cart(xyz, lattice) if direct else _vscale(xyz, factor)
            flags = tuple(fields[3:6]) if selective and len(fields) >= 6 else ()
            atoms.append(Atom(symbol, cart, flags, index))
            cursor += 1
            index += 1
    return Structure(lines[0].strip(), lattice, species, atoms, selective)


def _present_species(atoms: Iterable[Atom]) -> tuple[str, ...]:
    """Return the generated POSCAR/POTCAR order: always alphabetical."""
    return tuple(sorted({atom.symbol for atom in atoms}))


def _formula(atoms: Iterable[Atom]) -> str:
    atoms = list(atoms)
    return "".join(f"{s}{sum(a.symbol == s for a in atoms)}" for s in sorted({a.symbol for a in atoms}))


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_poscar(path: Path, structure: Structure, lattice: Lattice, atoms: list[Atom]) -> None:
    species = _present_species(atoms)
    counts = [sum(a.symbol == s for a in atoms) for s in species]
    ordered = [a for s in species for a in atoms if a.symbol == s]
    lines = [
        " ".join(species), "1.0",
        *("  " + "  ".join(f"{x:20.14f}" for x in v) for v in lattice),
        "  " + "  ".join(species), "  " + "  ".join(map(str, counts)),
    ]
    if structure.selective:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for atom in ordered:
        frac = _cart_to_frac(atom.cart, lattice)
        line = "  " + "  ".join(f"{x:20.14f}" for x in frac)
        if structure.selective:
            line += "  " + "  ".join(atom.flags or ("T", "T", "T"))
        lines.append(line)
    _atomic_text(path, "\n".join(lines) + "\n")


def _auto_plane(z_values: list[float], min_fraction: float) -> tuple[float, float]:
    z = sorted(z_values)
    n = len(z)
    min_side = max(1, math.ceil(n * min_fraction))
    choices = []
    for i in range(min_side, n - min_side + 1):
        gap = z[i] - z[i - 1]
        balance = 4 * i * (n - i) / (n * n)
        choices.append((gap * balance, gap, (z[i] + z[i - 1]) / 2))
    if not choices:
        raise ValueError("Too few atoms for automatic splitting")
    _, gap, plane = max(choices)
    return plane, gap


def _split_potcar(path: Path, expected: int) -> list[bytes]:
    data = path.read_bytes()
    marker = b"End of Dataset"
    pieces = data.split(marker)
    blocks = [piece + marker + b"\n" for piece in pieces[:-1]]
    if pieces[-1].strip():
        blocks.append(pieces[-1])
    if len(blocks) != expected:
        if expected == 1:
            return [data]
        raise ValueError(f"Expected {expected} datasets in POTCAR, found {len(blocks)}")
    return blocks


def _subset_potcar(path: Path, all_species: tuple[str, ...], blocks: list[bytes], wanted: tuple[str, ...]) -> None:
    mapping = dict(zip(all_species, blocks, strict=True))
    path.write_bytes(b"".join(mapping[s] for s in wanted))


def _incar_tag(line: str) -> str | None:
    content = line.split("#", 1)[0].strip()
    if "=" not in content:
        return None
    tag = content.split("=", 1)[0].strip().upper()
    return tag if tag.replace("_", "").isalnum() else None


def _adapt_incar(
    base: str,
    overrides: list[tuple[str, str]],
    remove: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
) -> str:
    """Preserve the user's INCAR while replacing task-critical tags once."""
    changed = {tag.upper() for tag, _ in overrides}
    removed = {tag.upper() for tag in (remove or set())}
    kept = []
    for line in base.splitlines():
        tag = _incar_tag(line)
        if tag in changed | removed:
            continue
        if tag is not None and any(tag.startswith(prefix.upper()) for prefix in remove_prefixes):
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend(["", "# Automatically set for the InterfaceForge adhesion workflow"])
    kept.extend(f"{tag:<15} = {value}" for tag, value in overrides)
    return "\n".join(kept) + "\n"


def _mlff_relax_incar(base: str) -> str:
    return _adapt_incar(
        base,
        [
            ("ML_LMLFF", ".TRUE."), ("ML_MODE", "run"),
            ("IBRION", "2"), ("NSW", "99"), ("POTIM", "0.20"),
            ("EDIFFG", "-0.02"), ("ISIF", "2"), ("ISYM", "0"),
            ("ML_OUTPUT_MODE", "1"), ("ML_OUTBLOCK", "1"),
            ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("LVHAR", ".FALSE."),
        ],
    )


def _static_incar(base: str, distance: float) -> str:
    return _adapt_incar(
        base,
        [
            ("SYSTEM", f"adhesion rigid separation {distance:.2f} A"),
            ("ML_LMLFF", ".TRUE."), ("ML_MODE", "run"),
            ("IBRION", "-1"), ("NSW", "1"), ("ISIF", "2"), ("ISYM", "0"),
            ("ML_OUTPUT_MODE", "1"), ("ML_OUTBLOCK", "1"),
            ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("LVHAR", ".FALSE."),
        ],
        remove={"EDIFFG", "POTIM"},
    )


def _dft_relax_incar(base: str) -> str:
    return _adapt_incar(
        base,
        [
            ("IBRION", "2"), ("NSW", "99"),
            ("EDIFF", "1E-6"), ("EDIFFG", "-0.02"),
            ("ISIF", "2"), ("ISYM", "0"),
            ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("LVHAR", ".FALSE."),
        ],
        remove_prefixes=("ML_",),
    )


def _dft_static_incar(base: str, distance: float) -> str:
    return _adapt_incar(
        base,
        [
            ("SYSTEM", f"adhesion DFT rigid separation {distance:.2f} A"),
            ("IBRION", "-1"), ("NSW", "0"),
            ("EDIFF", "1E-6"), ("ISIF", "2"), ("ISYM", "0"),
            ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("LVHAR", ".FALSE."),
        ],
        remove={"EDIFFG", "POTIM"},
        remove_prefixes=("ML_",),
    )


def _resolve_file(source: Path, requested: str | None, default: str) -> Path:
    path = Path(requested) if requested else source / default
    if requested and not path.is_absolute():
        path = source / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _find_launcher(source: Path, requested: str | None) -> Path | None:
    """Find the launcher to propagate into every generated run directory.

    Mirrors the preference order ``resolve_launcher`` in ``vasp.py`` already
    uses at submission time: an explicit name if given (required to exist),
    otherwise ``runvasp.sh`` then ``run.slurm`` in the source directory if
    present. Returns ``None`` when auto-detecting and neither exists; a
    launcher is convenient but was never required to prepare calculations.
    """
    if requested:
        path = (source / requested).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for name in ("runvasp.sh", "run.slurm"):
        path = source / name
        if path.is_file():
            return path
    return None


def _make_run_dir(
    path: Path, incar_text: str, kpoints: Path, model: Path | None, launcher: Path | None
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    _atomic_text(path / "INCAR", incar_text)
    shutil.copy2(kpoints, path / "KPOINTS")
    if model is not None:
        os.link(model, path / "ML_FF", follow_symlinks=True)
        source_stat = model.stat()
        linked_stat = (path / "ML_FF").stat()
        if (source_stat.st_dev, source_stat.st_ino) != (linked_stat.st_dev, linked_stat.st_ino):
            raise SafetyError(f"Hard-link verification failed for {path / 'ML_FF'}")
    if launcher is not None:
        shutil.copy2(launcher, path / launcher.name)


def prepare_adhesion(
    interface_dir: str | Path,
    *,
    method: str = "mlff",
    structure: str | None = None,
    incar: str | None = None,
    curve_incar: str | None = None,
    kpoints: str | None = None,
    potcar: str | None = None,
    z_plane: float | None = None,
    guard: float = 0.20,
    min_side_fraction: float = 0.10,
    lower_name: str = "lower",
    upper_name: str = "upper",
    distances: Iterable[float] = (0.5, 1, 2, 3, 4, 6, 8),
    output_dir: str | Path | None = None,
    launcher: str | None = None,
    propagate_launcher: bool = True,
) -> dict[str, Any]:
    """Prepare a work-of-adhesion calculation tree from a reference interface run.

    The reference directory (POSCAR/CONTCAR, KPOINTS, POTCAR, and an
    appropriate INCAR; MLFF mode additionally requires ML_FF) is never
    modified. The generated sibling tree keeps ``reference`` at its head as a
    relative symlink (the zero-separation point), creates relaxed
    isolated-slab inputs for each fragment, and creates static
    positive-separation inputs (a rigid separation curve). In MLFF mode every
    generated run receives a verified hard link to the reference ML_FF: VASP
    sees a regular file, while all copies share one inode and consume no
    additional model storage. DFT mode removes ML tags and creates no ML_FF.
    This function never launches VASP.

    When ``propagate_launcher`` is true (the default), the same launcher
    ``iface vasp submit`` would pick for the reference directory
    (``runvasp.sh``, else ``run.slurm``; or the explicit ``launcher`` name)
    is copied into every generated slab and rigid-curve directory, so each
    is independently submittable. Set ``propagate_launcher=False`` to skip
    this even if a launcher is present.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown adhesion method {method!r}; choose one of {METHODS}")
    source = Path(interface_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not 0 <= min_side_fraction < 0.5 or guard < 0:
        raise ValueError("Invalid min-side fraction or guard distance")
    if lower_name == upper_name:
        raise ValueError("Fragment names must differ")

    if structure:
        structure_path = _resolve_file(source, structure, "CONTCAR")
    else:
        structure_path = (source / ("CONTCAR" if (source / "CONTCAR").is_file() else "POSCAR")).resolve()
        if not structure_path.is_file():
            raise FileNotFoundError("Neither CONTCAR nor POSCAR exists")
    if incar:
        incar_path = _resolve_file(source, incar, "INCAR_MLFF_RELAX")
    else:
        preferred_incar = (
            "INCAR_MLFF_RELAX"
            if method == "mlff" and (source / "INCAR_MLFF_RELAX").is_file()
            else "INCAR"
        )
        incar_path = _resolve_file(source, None, preferred_incar)
    base_incar = incar_path.read_text(encoding="utf-8")
    kpoints_path = _resolve_file(source, kpoints, "KPOINTS")
    potcar_path = _resolve_file(source, potcar, "POTCAR")
    model = _resolve_file(source, None, "ML_FF") if method == "mlff" else None
    curve_incar_path = _resolve_file(source, curve_incar, "INCAR") if curve_incar else None
    curve_incar_text = curve_incar_path.read_text(encoding="utf-8") if curve_incar_path else None
    launcher_path = _find_launcher(source, launcher) if propagate_launcher else None

    if output_dir is None:
        output = source.parent / f"{source.name}_adhesion_{method}"
    else:
        candidate = Path(output_dir)
        output = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    output = output.resolve()
    if output.exists():
        raise SafetyError(f"{output} already exists; existing calculations are never overwritten")
    output.mkdir(parents=True)
    reference_link = output / "reference"
    reference_link.symlink_to(os.path.relpath(source, output), target_is_directory=True)
    if model is not None:
        hardlink_probe = output / ".ML_FF_hardlink_test"
        try:
            os.link(model, hardlink_probe, follow_symlinks=True)
            model_stat = model.stat()
            probe_stat = hardlink_probe.stat()
            if (model_stat.st_dev, model_stat.st_ino) != (probe_stat.st_dev, probe_stat.st_ino):
                raise SafetyError("The filesystem did not create a true ML_FF hard link")
        finally:
            if hardlink_probe.exists():
                hardlink_probe.unlink()

    parsed = _read_vasp(structure_path)
    if z_plane is None:
        plane, detected_gap = _auto_plane([a.cart[2] for a in parsed.atoms], min_side_fraction)
        plane_source = "balanced_internal_gap"
    else:
        plane, detected_gap, plane_source = z_plane, None, "explicit"
    nearest = min(parsed.atoms, key=lambda a: abs(a.cart[2] - plane))
    nearest_distance = abs(nearest.cart[2] - plane)
    if nearest_distance <= guard:
        raise ValueError(
            f"Atom {nearest.original_index + 1} ({nearest.symbol}) is only "
            f"{nearest_distance:.4f} A from the cut"
        )
    lower = [a for a in parsed.atoms if a.cart[2] < plane]
    upper = [a for a in parsed.atoms if a.cart[2] > plane]
    if not lower or not upper or len(lower) + len(upper) != len(parsed.atoms):
        raise ValueError("Cut did not produce two complete fragments")

    a, b, c = parsed.lattice
    normal_raw = _cross(a, b)
    area = _norm(normal_raw)
    normal = _vscale(normal_raw, 1 / area)
    if _dot(c, normal) < 0:
        normal = _vscale(normal, -1)

    distance_values = sorted(set(distances))
    if any(d < 0 for d in distance_values):
        raise ValueError("Separation distances must be nonnegative")
    positive_distances = [d for d in distance_values if d > 0.0]

    blocks = _split_potcar(potcar_path, len(parsed.species))
    slab_records = []
    for name, atoms in ((lower_name, lower), (upper_name, upper)):
        run = output / "slabs" / name
        relax_text = _mlff_relax_incar(base_incar) if method == "mlff" else _dft_relax_incar(base_incar)
        _make_run_dir(run, relax_text, kpoints_path, model, launcher_path)
        _write_poscar(run / "POSCAR", parsed, parsed.lattice, atoms)
        species = _present_species(atoms)
        _subset_potcar(run / "POTCAR", parsed.species, blocks, species)
        slab_records.append(
            {
                "name": name,
                "formula": _formula(atoms),
                "natoms": len(atoms),
                "potcar_order": species,
                "directory": str(run.relative_to(output)),
            }
        )

    curve_records = []
    for distance in positive_distances:
        run = output / "rigid_curve" / f"sep_{distance:06.2f}_A"
        if curve_incar_text is not None:
            curve_text = curve_incar_text
        elif method == "mlff":
            curve_text = _static_incar(base_incar, distance)
        else:
            curve_text = _dft_static_incar(base_incar, distance)
        _make_run_dir(run, curve_text, kpoints_path, model, launcher_path)
        full_species = _present_species(parsed.atoms)
        _subset_potcar(run / "POTCAR", parsed.species, blocks, full_species)
        lattice: Lattice = (a, b, _vadd(c, _vscale(normal, distance)))
        shifted = [Atom(x.symbol, _vadd(x.cart, _vscale(normal, distance)), x.flags, x.original_index) for x in upper]
        _write_poscar(run / "POSCAR", parsed, lattice, lower + shifted)
        curve_records.append(
            {
                "separation_A": distance,
                "directory": str(run.relative_to(output)),
                "cell_normal_A": _dot(lattice[2], normal),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "reference_directory": str(source),
        "reference_link": "reference",
        "reference_is_zero_separation": True,
        "structure": str(structure_path),
        "split_plane_z_A": plane,
        "plane_source": plane_source,
        "detected_gap_A": detected_gap,
        "nearest_atom_to_plane_A": nearest_distance,
        "interface_area_A2": area,
        "one_interface_J_m2_per_eV": EV_A2_TO_J_M2 / area,
        "parent_formula": _formula(parsed.atoms),
        "parent_natoms": len(parsed.atoms),
        "source_potcar_order": parsed.species,
        "generated_parent_poscar_potcar_order": _present_species(parsed.atoms),
        "slabs": slab_records,
        "rigid_curve": curve_records,
        "model_storage": "verified hard links to reference/ML_FF" if model is not None else "not used for DFT",
        "launcher": launcher_path.name if launcher_path is not None else None,
        "notes": [
            "The reference directory is the zero-separation point and is not modified.",
            "Rigid curve is static unless curve_incar is supplied.",
            "Upper slab and c vector move by the same distance, preserving outer vacuum.",
            "No calculations were launched.",
            (
                f"{launcher_path.name} was copied into every slab and rigid-curve directory "
                "from the reference; submit each independently."
                if launcher_path is not None
                else "No launcher (runvasp.sh or run.slurm) was found in the reference "
                "directory; none was propagated."
            ),
            "Work of adhesion W_ad = (E_lower_slab + E_upper_slab - E_reference) / interface_area_A2, "
            "using each run's converged energy once VASP has completed; multiply by "
            "one_interface_J_m2_per_eV to convert eV/A^2 to J/m^2.",
        ],
    }
    manifest_path = output / "manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    manifest["output_directory"] = str(output)
    manifest["manifest"] = str(manifest_path)
    return manifest
