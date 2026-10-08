"""Dependency-free readers for the VASP inputs a density initializer consumes.

Nothing here writes a file.  Every value an initializer needs (ion order,
signed ``MAGMOM``, ``NELECT``, ``LMAXMIX``, the FFT grid) is read from the
calculation directory itself so that the authoritative source is always the
InterfaceForge/user-prepared ``INCAR``/``POSCAR``/``POTCAR``, never a default
chosen by an external package.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ..errors import SafetyError
from ..vasp import _poscar_elements, _poscar_geometry_sha256, _poscar_layout_indices, parse_incar

REQUIRED_INPUTS = ("INCAR", "POSCAR", "POTCAR", "KPOINTS")

_NG_FINE = re.compile(r"NGXF\s*=\s*(\d+)\s+NGYF\s*=\s*(\d+)\s+NGZF\s*=\s*(\d+)")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def missing_inputs(run: Path) -> list[str]:
    """Required inputs that are absent or empty.

    ``KPOINTS`` may be omitted only when the INCAR sets ``KSPACING`` (VASP
    then generates the mesh itself).
    """

    missing = [name for name in REQUIRED_INPUTS if not nonempty(run / name)]
    if "KPOINTS" in missing and "KSPACING" in parse_incar(run / "INCAR"):
        missing.remove("KPOINTS")
    return missing


def input_hashes(run: Path, names: tuple[str, ...] = REQUIRED_INPUTS) -> dict[str, str | None]:
    return {name: sha256_file(run / name) if (run / name).is_file() else None for name in names}


# ----------------------------------------------------------------------------
# POSCAR
# ----------------------------------------------------------------------------


def poscar_species(poscar: Path) -> list[str]:
    """One element symbol per ion, in POSCAR (and therefore MAGMOM/CHGCAR) order."""

    elements = _poscar_elements(poscar)
    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    count_index, _, _ = _poscar_layout_indices(lines, poscar)
    counts = [int(token) for token in lines[count_index].split()]
    return [symbol for symbol, count in zip(elements, counts, strict=True) for _ in range(count)]


def poscar_lattice(poscar: Path) -> list[list[float]]:
    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 5:
        raise SafetyError(f"POSCAR is too short: {poscar}")
    scale = float(lines[1].split()[0])
    rows = [[float(value) for value in lines[index].split()[:3]] for index in (2, 3, 4)]
    if scale < 0:  # negative scale = target volume; lattice vectors used as given
        return rows
    return [[scale * value for value in row] for row in rows]


def structure_identity(poscar: Path) -> dict[str, Any]:
    species = poscar_species(poscar)
    return {
        "poscar_sha256": sha256_file(poscar),
        "geometry_sha256": _poscar_geometry_sha256(poscar),
        "n_ions": len(species),
        "species_order": list(dict.fromkeys(species)),
        "composition": {symbol: species.count(symbol) for symbol in dict.fromkeys(species)},
    }


# ----------------------------------------------------------------------------
# INCAR: MAGMOM, spin, LMAXMIX
# ----------------------------------------------------------------------------


def expand_vasp_list(value: str) -> list[float]:
    """Expand VASP ``N*value`` list shorthand, preserving sign and order exactly."""

    out: list[float] = []
    for token in value.replace(",", " ").split():
        if "*" in token:
            count, _, item = token.partition("*")
            try:
                repeat = int(count)
                number = float(item)
            except ValueError as exc:
                raise SafetyError(f"Cannot parse VASP list token {token!r}") from exc
            if repeat < 0:
                raise SafetyError(f"Negative repeat count in VASP list token {token!r}")
            out.extend([number] * repeat)
        else:
            try:
                out.append(float(token))
            except ValueError as exc:
                raise SafetyError(f"Cannot parse VASP list token {token!r}") from exc
    return out


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().strip(".").upper() in {"TRUE", "T"}


def magnetic_settings(incar: dict[str, str], n_ions: int) -> dict[str, Any]:
    """The authoritative collinear magnetic configuration of one run.

    Raises on non-collinear/SOC runs (no initializer here supports them) and
    on a ``MAGMOM`` whose length does not match the ion count.
    """

    if _truthy(incar.get("LNONCOLLINEAR")) or _truthy(incar.get("LSORBIT")):
        raise SafetyError(
            "Non-collinear / spin-orbit INCARs are not supported by density "
            "initialization; run them with the standard start"
        )
    ispin = int(float(incar.get("ISPIN", "1")))
    raw = incar.get("MAGMOM")
    magmom = expand_vasp_list(raw) if raw is not None else None
    if magmom is not None and len(magmom) != n_ions:
        raise SafetyError(f"INCAR MAGMOM has {len(magmom)} entries but POSCAR has {n_ions} ions")
    nonzero = [m for m in (magmom or []) if abs(m) > 1e-8]
    signs = {m > 0 for m in nonzero}
    if ispin != 2:
        order = "non-spin-polarized"
    elif magmom is None:
        order = "vasp-default (MAGMOM absent: NIONS*1.0)"
    elif not nonzero:
        order = "zero"
    elif len(signs) == 1:
        order = "sign-uniform"
    else:
        order = "mixed-sign"
    return {
        "ispin": ispin,
        "incar_magmom_raw": raw,
        "incar_magmom": magmom,
        "net_moment": sum(magmom) if magmom is not None else None,
        "order": order,
        "mixed_sign": order == "mixed-sign",
        "nupdown": incar.get("NUPDOWN"),
    }


def lmaxmix(incar: dict[str, str]) -> int:
    return int(float(incar.get("LMAXMIX", "2")))


# ----------------------------------------------------------------------------
# POTCAR: TITEL / ZVAL
# ----------------------------------------------------------------------------


def potcar_entries(potcar: Path) -> list[dict[str, Any]]:
    """``[{titel, symbol, element, zval, projector_l}]`` in POTCAR order.

    ``projector_l`` lists one angular momentum per PAW projector, read from the
    ``Non local Part`` blocks (each opens with ``l`` and the projector count
    for that ``l``).  It fixes the size/layout of the augmentation-occupancy
    block VASP expects per ion in a CHGCAR.
    """

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    expect_nonlocal = False
    for line in potcar.read_text(encoding="utf-8", errors="ignore").splitlines():
        if expect_nonlocal:
            expect_nonlocal = False
            tokens = line.split()
            if current is not None and len(tokens) >= 2:
                try:
                    current["projector_l"].extend([int(tokens[0])] * int(tokens[1]))
                except ValueError:
                    pass
            continue
        if "TITEL" in line and "=" in line:
            titel = line.split("=", 1)[1].strip()
            parts = titel.split()
            symbol = parts[1] if len(parts) > 1 else parts[0]
            current = {
                "titel": titel,
                "symbol": symbol,
                "element": symbol.split("_")[0],
                "zval": None,
                "projector_l": [],
            }
            entries.append(current)
        elif current is not None and current["zval"] is None and "ZVAL" in line:
            match = re.search(r"ZVAL\s*=\s*([0-9.]+)", line)
            if match:
                current["zval"] = float(match.group(1))
        elif "Non local Part" in line:
            expect_nonlocal = True
    return entries


def nelect(run: Path, incar: dict[str, str], species: list[str]) -> tuple[float, str]:
    """Valence electron count: INCAR ``NELECT`` > POTCAR ZVAL x POSCAR counts."""

    if "NELECT" in incar:
        return float(incar["NELECT"]), "INCAR NELECT"
    entries = potcar_entries(run / "POTCAR")
    elements = list(dict.fromkeys(species))
    if len(entries) != len(elements):
        raise SafetyError(
            f"POTCAR has {len(entries)} datasets but POSCAR has {len(elements)} species"
        )
    for element, entry in zip(elements, entries, strict=True):
        if entry["element"] != element:
            raise SafetyError(
                f"POTCAR order ({[e['element'] for e in entries]}) does not match POSCAR ({elements})"
            )
        if entry["zval"] is None:
            raise SafetyError(f"POTCAR dataset {entry['titel']!r} has no ZVAL")
    zval = {element: entry["zval"] for element, entry in zip(elements, entries, strict=True)}
    return float(sum(zval[symbol] for symbol in species)), "POTCAR ZVAL"


# ----------------------------------------------------------------------------
# FFT grid (NGXF/NGYF/NGZF)
# ----------------------------------------------------------------------------

# Tags that decide VASP's fine FFT grid.  A grid read from another run is only
# reused when all of these agree and the lattice is identical.
GRID_TAGS = ("ENCUT", "PREC", "ENAUG", "NGX", "NGY", "NGZ", "NGXF", "NGYF", "NGZF", "ADDGRID")


def grid_from_incar(incar: dict[str, str]) -> tuple[int, int, int] | None:
    if all(tag in incar for tag in ("NGXF", "NGYF", "NGZF")):
        return tuple(int(float(incar[tag])) for tag in ("NGXF", "NGYF", "NGZF"))  # type: ignore[return-value]
    return None


def grid_from_outcar(outcar: Path) -> tuple[int, int, int] | None:
    """First ``NGXF= NGYF= NGZF=`` triple in an OUTCAR (the run's fine grid)."""

    with outcar.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            match = _NG_FINE.search(line)
            if match:
                return tuple(int(group) for group in match.groups())  # type: ignore[return-value]
            if index > 20000:
                break
    return None


def grid_from_chgcar(chgcar: Path) -> tuple[int, int, int] | None:
    """Grid line of a CHGCAR/LOCPOT: the integer triple after the coordinate block."""

    head: list[str] = []
    with chgcar.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(9):  # title..mode line (+ optional Selective dynamics)
            head.append(handle.readline().rstrip("\n"))
        try:
            _, ion_count, mode_index = _poscar_layout_indices(head, chgcar)
        except SafetyError:
            return None
        remaining = mode_index + 1 + ion_count - len(head)
        for _ in range(max(remaining, 0)):
            handle.readline()
        for _ in range(4):
            tokens = handle.readline().split()
            if len(tokens) == 3 and all(token.isdigit() for token in tokens):
                return tuple(int(token) for token in tokens)  # type: ignore[return-value]
    return None


def grid_from_file(path: Path) -> tuple[int, int, int] | None:
    name = path.name.upper()
    if "OUTCAR" in name:
        return grid_from_outcar(path)
    if "CHG" in name or "LOCPOT" in name:
        return grid_from_chgcar(path)
    return grid_from_outcar(path) or grid_from_chgcar(path)


def _same_lattice(left: Path, right: Path, tol: float = 1e-6) -> bool:
    a = poscar_lattice(left)
    b = poscar_lattice(right)
    return all(abs(x - y) <= tol for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True))


def reusable_grid(
    target_run: Path,
    reference_run: Path,
    *,
    reference_structure: str = "POSCAR",
    target_incar: dict[str, str] | None = None,
    target_poscar: Path | None = None,
) -> tuple[tuple[int, int, int] | None, str]:
    """Reuse ``reference_run/OUTCAR``'s fine grid only when it provably applies.

    It applies when every grid-deciding INCAR tag agrees and the reference
    OUTCAR's input lattice (its ``POSCAR``) equals the target lattice.  The
    target may be given as parsed tags + a structure file when it has not
    been written yet.
    """

    outcar = reference_run / "OUTCAR"
    if not nonempty(outcar):
        return None, f"{outcar} missing"
    target_tags = target_incar if target_incar is not None else parse_incar(target_run / "INCAR")
    reference_incar = parse_incar(reference_run / "INCAR")
    differing = [
        tag
        for tag in GRID_TAGS
        if str(target_tags.get(tag, "")).upper() != str(reference_incar.get(tag, "")).upper()
    ]
    if differing:
        return None, f"grid-deciding INCAR tags differ from {reference_run}: {', '.join(differing)}"
    reference_poscar = reference_run / reference_structure
    structure = target_poscar if target_poscar is not None else target_run / "POSCAR"
    if not nonempty(reference_poscar) or not _same_lattice(structure, reference_poscar):
        return None, f"lattice differs from {reference_poscar}"
    grid = grid_from_outcar(outcar)
    if grid is None:
        return None, f"no NGXF/NGYF/NGZF line in {outcar}"
    return grid, f"OUTCAR {outcar} (same grid tags and lattice)"
