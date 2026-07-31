"""VASP and VASP-MLFF preparation, continuation, and recovery helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .errors import SafetyError

_INCAR_LINE = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")
_ARCHIVE_FILES = (
    "run.slurm",
    "runvasp.sh",
    "INCAR",
    "KPOINTS",
    "POTCAR",
    "POSCAR",
    "CONTCAR",
    "XDATCAR",
    "XDATCAR_FINAL",
    "OSZICAR",
    "OUTCAR",
    "ML_LOGFILE",
    "REPORT",
    "ML_AB",
    "ML_ABN",
    "ML_FF",
    "ML_FFN",
    "vasprun.xml",
)

INCAR_PRESETS = ("static", "relax", "md", "dos")


def parse_incar(path: str | Path) -> dict[str, str]:
    """Parse the last active value of each INCAR tag."""

    parsed: dict[str, str] = {}
    incar = Path(path)
    if not incar.is_file():
        return parsed
    for raw in incar.read_text(encoding="utf-8", errors="ignore").splitlines():
        active = re.split(r"[!#]", raw, maxsplit=1)[0]
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", active)
        if match:
            parsed[match.group(1).upper()] = match.group(2).strip()
    return parsed


def update_incar(
    path: str | Path,
    changes: Mapping[str, Any],
    *,
    delete: Iterable[str] = (),
    create: bool = False,
) -> Path:
    """Update tags while preserving unrelated comments and ordering."""

    incar = Path(path)
    if not incar.exists() and not create:
        raise FileNotFoundError(incar)
    original = incar.read_text(encoding="utf-8", errors="ignore") if incar.exists() else ""
    normalized = {str(key).upper(): str(value) for key, value in changes.items()}
    deleted = {str(key).upper() for key in delete}
    found: set[str] = set()
    output: list[str] = []

    for line in original.splitlines(keepends=True):
        match = _INCAR_LINE.match(line)
        if not match:
            output.append(line)
            continue
        key = match.group(2).upper()
        if key in deleted:
            continue
        if key in normalized:
            ending = match.group(5) or "\n"
            output.append(f"{match.group(1)}{key}{match.group(3)}{normalized[key]}{ending}")
            found.add(key)
        else:
            output.append(line)

    missing = [key for key in normalized if key not in found]
    if missing:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        if output and output[-1].strip():
            output.append("\n")
        output.extend(f"{key} = {normalized[key]}\n" for key in missing)

    temporary = incar.with_suffix(incar.suffix + ".tmp")
    temporary.write_text("".join(output), encoding="utf-8")
    temporary.replace(incar)
    return incar


def incar_preset(
    name: str,
    *,
    temperature: float = 300.0,
    nsw: int = 3000,
    potim: float = 1.0,
) -> tuple[dict[str, Any], set[str]]:
    """Return a small, explicit ionic/electronic-control preset.

    The presets intentionally avoid system-dependent convergence choices such
    as ENCUT, KSPACING, EDIFF, spin, and dispersion corrections.
    """

    if name == "static":
        return (
            {"IBRION": -1, "NSW": 0},
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM", "EDIFFG"},
        )
    if name == "relax":
        return (
            {"IBRION": 2, "NSW": nsw, "ISIF": 3, "EDIFFG": -0.02},
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM"},
        )
    if name == "md":
        return (
            {
                "IBRION": 0,
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": 2,
                "SMASS": 1.0,
                "TEBEG": temperature,
                "TEEND": temperature,
                "ISIF": 2,
            },
            {"EDIFFG"},
        )
    if name == "dos":
        return (
            {
                "IBRION": -1,
                "NSW": 0,
                "ICHARG": 11,
                "LORBIT": 11,
                "NEDOS": 2000,
                "ISMEAR": -5,
            },
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM", "EDIFFG"},
        )
    raise ValueError(f"Unknown INCAR preset {name!r}; choose from {INCAR_PRESETS}")


def apply_incar_preset(
    path: str | Path,
    preset: str,
    *,
    temperature: float = 300.0,
    nsw: int = 3000,
    potim: float = 1.0,
    create: bool = False,
) -> dict[str, Any]:
    """Apply one conservative preset without overwriting unrelated settings."""

    changes, delete = incar_preset(
        preset,
        temperature=temperature,
        nsw=nsw,
        potim=potim,
    )
    output = update_incar(path, changes, delete=delete, create=create)
    return {
        "incar": str(output.resolve()),
        "preset": preset,
        "changes": changes,
        "removed_tags": sorted(delete),
    }


def stage_tags(stage: str, *, temperature: float, nsw: int, potim: float) -> tuple[dict[str, Any], set[str]]:
    """Return conservative VASP MLFF tags for one stage."""

    common = {
        "ML_LMLFF": ".TRUE.",
        "ML_WTSIF": "1E-10",
        "ISYM": "0",
    }
    if stage == "train":
        return (
            {
                **common,
                "ML_MODE": "train",
                "IBRION": "0",
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": "2",
                "SMASS": "1.0",
                "TEBEG": temperature,
                "TEEND": temperature,
                "ISIF": "2",
            },
            {"ML_ISTART", "ML_ESTBLOCK"},
        )
    if stage == "refit":
        return ({**common, "ML_MODE": "refit"}, {"ML_ISTART", "IBRION", "NSW", "ML_ESTBLOCK"})
    if stage == "stability":
        return (
            {
                **common,
                "ML_MODE": "run",
                "IBRION": "0",
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": "2",
                "SMASS": "1.0",
                "TEBEG": temperature,
                "TEEND": temperature,
                "ISIF": "2",
                "ML_ESTBLOCK": "50",
            },
            {
                "ML_ISTART",
                "ML_MB",
                "ML_MCONF",
                "ML_LBASIS_DISCARD",
                "ML_CDOUB",
                "ML_CSLOPE",
                "ML_ICRITERIA",
                "ML_NMDINT",
            },
        )
    if stage == "validation_dft":
        return (
            {
                "ML_LMLFF": ".FALSE.",
                "IBRION": "-1",
                "NSW": "0",
            },
            {"ML_MODE", "ML_ISTART", "ML_ESTBLOCK"},
        )
    if stage == "validation_ml":
        return (
            {
                **common,
                "ML_MODE": "run",
                "IBRION": "-1",
                "NSW": "0",
                "ML_ESTBLOCK": "1",
            },
            {"ML_ISTART"},
        )
    raise ValueError(f"Unknown VASP stage: {stage}")


def require_files(folder: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (folder / name).is_file() or not (folder / name).stat().st_size]
    if missing:
        raise SafetyError(f"Missing required files in {folder}: {', '.join(missing)}")


def archive_run(folder: str | Path, operation: str) -> Path:
    """Create a recoverable archive before mutating a run folder."""

    run = Path(folder).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = run / ".interfaceforge" / "archive" / f"{operation}_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    copied = 0
    for name in _ARCHIVE_FILES:
        source = run / name
        if source.is_file():
            shutil.copy2(source, archive / name)
            copied += 1
    for pattern in ("slurm-*.out", "slurm-*.err", "vasp.*.out", "vasp.*.err"):
        for source in run.glob(pattern):
            if source.is_file():
                shutil.copy2(source, archive / source.name)
                copied += 1
    if copied == 0:
        raise SafetyError(f"No run state was available to archive in {run}")
    return archive


def _clean_outputs(folder: Path) -> None:
    for name in (
        "WAVECAR",
        "CHGCAR",
        "EIGENVAL",
        "DOSCAR",
        "PROCAR",
        "PCDAT",
        "REPORT",
        "vasprun.xml",
        "OSZICAR",
        "OUTCAR",
        "ML_LOGFILE",
    ):
        (folder / name).unlink(missing_ok=True)


def _continue_source(folder: Path) -> Path:
    source = folder / "ML_ABN"
    if source.is_file() and source.stat().st_size:
        return source
    source = folder / "ML_AB"
    if source.is_file() and source.stat().st_size:
        return source
    raise SafetyError(f"Neither ML_ABN nor ML_AB is available in {folder}")


def prepare_recovery(
    folder: str | Path,
    operation: str,
    *,
    temperature: float | None = None,
    nsw: int | None = None,
    ml_mb: int | None = None,
    ml_mconf: int | None = None,
    force_expand: bool = False,
) -> dict[str, Any]:
    """Safely mutate one VASP MLFF folder for a well-defined recovery operation."""

    run = Path(folder).resolve()
    require_files(run, ("INCAR", "POTCAR", "KPOINTS"))
    if operation not in {"continue", "expand", "refit", "stability"}:
        raise ValueError(f"Unknown recovery operation: {operation}")

    if operation in {"continue", "expand"}:
        require_files(run, ("CONTCAR",))
        source = _continue_source(run)
    elif operation == "refit":
        source = _continue_source(run)
    else:
        require_files(run, ("ML_FFN", "CONTCAR"))
        header = (run / "ML_FFN").open("rb").read(4096).decode("utf-8", errors="ignore")
        if not re.search(r"ML_LFAST.{0,20}(true|T)", header, re.I):
            raise SafetyError("ML_FFN does not report ML_LFAST=true")
        source = run / "ML_FFN"

    if operation == "expand":
        if ml_mb is None or ml_mb < 1:
            raise SafetyError("--ml-mb must be a positive integer for expansion")
        outcar_text = (run / "OUTCAR").read_text(errors="ignore") if (run / "OUTCAR").is_file() else ""
        capacity_stop = re.search(r"increase\s+ML_M(B|CONF)|ML_M(B|CONF).*(too small|exceed)", outcar_text, re.I)
        if not capacity_stop and not force_expand:
            raise SafetyError(
                "OUTCAR does not show a recognized ML_MB/ML_MCONF capacity stop; "
                "use --force-expand only after manual confirmation"
            )

    archive = archive_run(run, operation)
    if (run / "CONTCAR").is_file() and (run / "CONTCAR").stat().st_size:
        shutil.copy2(run / "CONTCAR", run / "POSCAR")
    _clean_outputs(run)

    if operation in {"continue", "expand", "refit"}:
        destination = run / "ML_AB"
        if source != destination:
            shutil.copy2(source, destination)
        for name in ("ML_ABN", "ML_FF", "ML_FFN"):
            (run / name).unlink(missing_ok=True)
    else:
        shutil.copy2(source, run / "ML_FF")

    current = parse_incar(run / "INCAR")
    selected_temperature = float(
        temperature if temperature is not None else current.get("TEBEG", 300)
    )
    selected_nsw = int(nsw if nsw is not None else current.get("NSW", 3000))
    selected_potim = float(current.get("POTIM", 1.0))

    if operation in {"continue", "expand"}:
        changes, delete = stage_tags(
            "train",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
        if operation == "expand":
            changes["ML_MB"] = int(ml_mb)
            if ml_mconf is not None:
                changes["ML_MCONF"] = int(ml_mconf)
            changes["ML_LBASIS_DISCARD"] = ".FALSE."
    elif operation == "refit":
        changes, delete = stage_tags(
            "refit",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
    else:
        changes, delete = stage_tags(
            "stability",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
    changes["ISTART"] = 0
    delete.add("ICHARG")
    update_incar(run / "INCAR", changes, delete=delete)

    return {
        "folder": str(run),
        "operation": operation,
        "archive": str(archive),
        "temperature": selected_temperature,
        "nsw": selected_nsw if operation != "refit" else None,
        "ml_mb": ml_mb,
        "ml_mconf": ml_mconf,
    }


def submit_run(folder: str | Path, launcher: str = "run.slurm") -> str:
    """Submit one prepared run and return the scheduler job id."""

    run = Path(folder).resolve()
    script = run / launcher
    if not script.is_file():
        raise FileNotFoundError(script)
    result = subprocess.run(
        ["sbatch", script.name],
        cwd=run,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _poscar_elements(poscar: Path) -> list[str]:
    """Read VASP 5+ species names from the canonical symbols line."""

    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise SafetyError(f"POSCAR is too short: {poscar}")
    symbols = lines[5].split()
    if not symbols or all(re.fullmatch(r"\d+", token) for token in symbols):
        raise SafetyError(
            "POSCAR does not contain a VASP 5+ element-symbol line. "
            "Convert it with `iface vasp geom convert` first."
        )
    counts = lines[6].split()
    if len(counts) != len(symbols) or not all(re.fullmatch(r"\d+", token) for token in counts):
        raise SafetyError(f"POSCAR element and count lines are inconsistent: {poscar}")
    return symbols


def assemble_potcar(
    poscar: str | Path,
    output: str | Path,
    *,
    pseudopotential_root: str | Path,
    mapping_file: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble POTCAR from an explicit licensed local pseudopotential tree."""

    poscar_path = Path(poscar).resolve()
    output_path = Path(output).resolve()
    root = Path(pseudopotential_root).expanduser().resolve()
    mapping_path = Path(mapping_file).resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing POTCAR: {output_path}")
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, Mapping):
        raise SafetyError(f"Invalid POTCAR map: {mapping_path}")
    elements = _poscar_elements(poscar_path)
    source_files: list[Path] = []
    variants: list[str] = []
    for element in elements:
        variant = str(mapping.get(element, "")).strip()
        if not variant:
            raise SafetyError(f"No POTCAR mapping for element {element}")
        source = root / variant / "POTCAR"
        if not source.is_file() or not source.stat().st_size:
            raise SafetyError(f"Missing licensed POTCAR source: {source}")
        source_files.append(source)
        variants.append(variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        for source in source_files:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, destination)
    temporary.replace(output_path)
    return {
        "output": str(output_path),
        "elements": elements,
        "variants": variants,
        "sources": [str(path) for path in source_files],
    }


def prepare_standard_restart(
    folder: str | Path,
    *,
    from_contcar: bool = True,
    clean_electronic: bool = False,
) -> dict[str, Any]:
    """Prepare an ordinary VASP restart without submitting or regenerating POTCAR."""

    run = Path(folder).resolve()
    require_files(run, ("INCAR", "POSCAR", "POTCAR", "KPOINTS"))
    if from_contcar:
        require_files(run, ("CONTCAR",))
    archive = archive_run(run, "vasp_restart")
    if from_contcar:
        shutil.copy2(run / "CONTCAR", run / "POSCAR")
    delete = {"ICHARG"}
    changes: dict[str, Any] = {}
    if clean_electronic:
        delete.add("ISTART")
        for name in ("WAVECAR", "CHG", "CHGCAR"):
            (run / name).unlink(missing_ok=True)
        changes["ISTART"] = 0
    update_incar(run / "INCAR", changes, delete=delete)
    return {
        "folder": str(run),
        "archive": str(archive),
        "from_contcar": from_contcar,
        "clean_electronic": clean_electronic,
        "submitted": False,
    }


def prepare_band_run(
    source_folder: str | Path,
    destination: str | Path,
    *,
    line_kpoints: str | Path,
    lmaxmix: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare a non-self-consistent line-mode band calculation."""

    source = Path(source_folder).resolve()
    target = Path(destination).resolve()
    kpoints = Path(line_kpoints).resolve()
    require_files(source, ("INCAR", "POSCAR", "POTCAR", "CHGCAR"))
    if not kpoints.is_file() or not kpoints.stat().st_size:
        raise SafetyError(f"Missing line-mode KPOINTS: {kpoints}")
    if target.exists() and any(target.iterdir()) and not force:
        raise SafetyError(f"Refusing to replace nonempty band directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("INCAR", "POSCAR", "POTCAR", "CHGCAR", "run.slurm"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)
    if (source / "CONTCAR").is_file() and (source / "CONTCAR").stat().st_size:
        shutil.copy2(source / "CONTCAR", target / "POSCAR")
    shutil.copy2(kpoints, target / "KPOINTS")
    update_incar(
        target / "INCAR",
        {
            "ICHARG": 11,
            "NSW": 0,
            "IBRION": -1,
            "ALGO": "Normal",
            "LREAL": "Auto",
            "LMAXMIX": int(lmaxmix),
            "ISMEAR": 0,
        },
        delete={"KPAR", "NCORE"},
    )
    return {"source": str(source), "destination": str(target), "kpoints": str(kpoints)}


def package_outputs(
    root: str | Path,
    output: str | Path,
    *,
    include_large: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Create a lightweight, non-destructive archive of reproducibility outputs."""

    source_root = Path(root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing archive: {output_path}")
    names = {
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "CONTCAR",
        "OSZICAR",
        "ML_LOGFILE",
        "run.slurm",
        "EIGENVAL",
        "DOSCAR",
    }
    if include_large:
        names.update({"OUTCAR", "vasprun.xml", "XDATCAR", "LOCPOT"})
    # POTCAR is intentionally excluded from a portable archive.
    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.name in names
        and ".interfaceforge/archive" not in path.as_posix()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(source_root))
    return {
        "root": str(source_root),
        "output": str(output_path),
        "files": len(files),
        "include_large": include_large,
        "potcar_excluded": True,
    }
