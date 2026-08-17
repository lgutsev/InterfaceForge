"""VASP and VASP-MLFF preparation, continuation, and recovery helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib import resources
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
MLFF_ACCURACY_PROFILES = ("accurate",)


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


def mlff_accuracy_profile_tags(profile: str, stage: str) -> dict[str, Any]:
    """Return opt-in VASP best-practice tags for accuracy-oriented MLFF stages."""

    if profile not in MLFF_ACCURACY_PROFILES:
        raise ValueError(
            f"Unknown MLFF accuracy profile {profile!r}; choose from {MLFF_ACCURACY_PROFILES}"
        )
    if stage == "train":
        return {
            "ML_IALGO_LINREG": "1",
            "ML_SION1": "0.3",
            "ML_MRB2": "12",
        }
    if stage == "refit":
        return {
            "ML_IALGO_LINREG": "4",
            "ML_SION1": "0.5",
            "ML_MRB2": "12",
            "ML_EPS_LOW": "1E-11",
        }
    return {}


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
    increase_eps_low: bool = False,
) -> dict[str, Any]:
    """Safely mutate one VASP MLFF folder for a well-defined recovery operation."""

    run = Path(folder).resolve()
    require_files(run, ("INCAR", "POTCAR", "KPOINTS"))
    if operation not in {"continue", "discard", "expand", "refit", "stability"}:
        raise ValueError(f"Unknown recovery operation: {operation}")
    if increase_eps_low and operation != "discard":
        raise SafetyError("ML_EPS_LOW adjustment is supported only with discard recovery")

    if operation in {"continue", "discard", "expand"}:
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

    if operation in {"discard", "expand"}:
        outcar_text = (run / "OUTCAR").read_text(errors="ignore") if (run / "OUTCAR").is_file() else ""
        capacity_stop = re.search(r"increase\s+ML_M(B|CONF)|ML_M(B|CONF).*(too small|exceed)", outcar_text, re.I)
        capacity_stop = capacity_stop or re.search(
            r"not enough storage reserved for local reference configurations",
            outcar_text,
            re.I,
        )
        if not capacity_stop and not force_expand:
            raise SafetyError(
                "OUTCAR does not show a recognized ML_MB/ML_MCONF capacity stop; "
                "use --force-expand only after manual confirmation"
            )
    if operation == "expand" and (ml_mb is None or ml_mb < 1):
        raise SafetyError("--ml-mb must be a positive integer for expansion")

    archive = archive_run(run, operation)
    if (run / "CONTCAR").is_file() and (run / "CONTCAR").stat().st_size:
        shutil.copy2(run / "CONTCAR", run / "POSCAR")
    _clean_outputs(run)

    if operation in {"continue", "discard", "expand", "refit"}:
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

    if operation in {"continue", "discard", "expand"}:
        changes, delete = stage_tags(
            "train",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
        if operation == "discard":
            changes["ML_LBASIS_DISCARD"] = ".TRUE."
            if increase_eps_low:
                try:
                    old_eps_low = float(current.get("ML_EPS_LOW", "1E-9"))
                except ValueError as error:
                    raise SafetyError("INCAR contains an invalid ML_EPS_LOW value") from error
                new_eps_low = old_eps_low * 10.0
                if not 0.0 < new_eps_low < 1.0e-7:
                    raise SafetyError(
                        "Tenfold ML_EPS_LOW increase must remain strictly below 1E-7; "
                        f"current value is {old_eps_low:g}"
                    )
                changes["ML_EPS_LOW"] = f"{new_eps_low:.0E}"
        elif operation == "expand":
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
        "ml_eps_low": parse_incar(run / "INCAR").get("ML_EPS_LOW"),
    }


def resolve_launcher(folder: str | Path, launcher: str | None = None) -> Path:
    """Resolve an explicit launcher or prefer the standalone VASP launcher."""

    run = Path(folder).resolve()
    if launcher:
        script = run / launcher
        if not script.is_file():
            raise FileNotFoundError(script)
        return script
    for name in ("runvasp.sh", "run.slurm"):
        script = run / name
        if script.is_file():
            return script
    raise FileNotFoundError(
        f"No VASP launcher found in {run}; expected runvasp.sh or run.slurm"
    )


def submit_run(
    folder: str | Path,
    launcher: str | None = None,
    *,
    potcar_root: str | Path | None = None,
    potcar_mapping: str | Path | None = None,
) -> str:
    """Submit one prepared run and return the scheduler job id."""

    run = Path(folder).resolve()
    ensure_run_potcar(
        run,
        pseudopotential_root=potcar_root,
        mapping_file=potcar_mapping,
    )
    script = resolve_launcher(run, launcher)
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
    """Read VASP 5+ species, with the legacy first-line convention as fallback."""

    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise SafetyError(f"POSCAR is too short: {poscar}")
    symbols = lines[5].split()
    if symbols and all(re.fullmatch(r"\d+", token) for token in symbols):
        counts = symbols
        symbols = lines[0].split()
    else:
        counts = lines[6].split()
    if not symbols or any(not re.fullmatch(r"[A-Z][a-z]?", token) for token in symbols):
        raise SafetyError(
            "POSCAR contains neither a valid VASP 5+ species line nor a legacy "
            "first-line element list"
        )
    if len(counts) != len(symbols) or not all(re.fullmatch(r"\d+", token) for token in counts):
        raise SafetyError(f"POSCAR element and count lines are inconsistent: {poscar}")
    return symbols


def assemble_potcar(
    poscar: str | Path,
    output: str | Path,
    *,
    pseudopotential_root: str | Path,
    mapping_file: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble POTCAR from an explicit licensed local pseudopotential tree."""

    poscar_path = Path(poscar).resolve()
    output_path = Path(output).resolve()
    root = Path(pseudopotential_root).expanduser().resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing POTCAR: {output_path}")
    if mapping_file is None:
        mapping_label = "built-in POTCAR_DEFS mapping"
        mapping_text = resources.files("interfaceforge").joinpath(
            "templates/potcar_pbe_54.yaml"
        ).read_text(encoding="utf-8")
    else:
        mapping_path = Path(mapping_file).resolve()
        mapping_label = str(mapping_path)
        mapping_text = mapping_path.read_text(encoding="utf-8")
    mapping = yaml.safe_load(mapping_text)
    if not isinstance(mapping, Mapping):
        raise SafetyError(f"Invalid POTCAR map: {mapping_label}")
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
        "mapping": mapping_label,
    }


def resolve_potcar_root(explicit: str | Path | None = None) -> Path:
    """Find the licensed local PBE PAW tree without bundling POTCAR data."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    if value := os.environ.get("IFACE_POTCAR_ROOT"):
        candidates.append(Path(value).expanduser())
    if value := os.environ.get("VASP_PP_PATH"):
        base = Path(value).expanduser()
        candidates.extend((base / "potpaw_PBE", base))
    candidates.append(Path.home() / "pot" / "potpaw_PBE")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise SafetyError(
        "POTCAR is missing and no licensed PBE PAW tree was found. "
        "Pass --potcar-root, set IFACE_POTCAR_ROOT, or set VASP_PP_PATH. "
        f"Searched: {searched}"
    )


def ensure_run_potcar(
    folder: str | Path,
    *,
    pseudopotential_root: str | Path | None = None,
    mapping_file: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a missing/empty run POTCAR from POSCAR before submission."""

    run = Path(folder).resolve()
    output = run / "POTCAR"
    if output.is_file() and output.stat().st_size:
        return {"status": "existing", "output": str(output)}
    require_files(run, ("POSCAR",))
    root = resolve_potcar_root(pseudopotential_root)
    payload = assemble_potcar(
        run / "POSCAR",
        output,
        pseudopotential_root=root,
        mapping_file=mapping_file,
        force=output.exists(),
    )
    payload["status"] = "generated"
    return payload


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


_MODEL_ARCHIVE_FILES = {
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "CONTCAR",
    "OSZICAR",
    "ML_LOGFILE",
    "REPORT",
    "run.slurm",
    "runvasp.sh",
    "XDATCAR_FINAL",
    "vasp_md_FINAL.dat",
    "ML_AB",
    "ML_ABN",
    "ML_FF",
    "ML_FFN",
}
_MODEL_ARCHIVE_LARGE_FILES = {"OUTCAR", "vasprun.xml", "XDATCAR", "LOCPOT"}
_MODEL_ARCHIVE_MANIFEST = "interfaceforge-model-archive.json"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_mlff_models(
    root: str | Path,
    output: str | Path,
    *,
    include_large: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Archive user-selected VASP-MLFF runs containing a nonempty ML_AB.

    Model presence is used only for discovery. The command intentionally does
    not claim that a model is scientifically validated; callers should point it
    at runs they have already accepted. POTCAR is never included.
    """

    source_root = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise SafetyError(f"Model archive root is not a directory: {source_root}")
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing archive: {output_path}")

    model_paths = sorted(
        path
        for path in source_root.rglob("ML_AB")
        if path.is_file()
        and not path.is_symlink()
        and path.stat().st_size
        and ".interfaceforge/archive" not in path.as_posix()
    )
    if not model_paths:
        raise SafetyError(f"No nonempty ML_AB files found below {source_root}")

    selected_names = set(_MODEL_ARCHIVE_FILES)
    if include_large:
        selected_names.update(_MODEL_ARCHIVE_LARGE_FILES)

    runs: list[dict[str, Any]] = []
    files_to_archive: list[tuple[Path, Path]] = []
    for model_path in model_paths:
        run = model_path.parent
        relative_run = run.relative_to(source_root)
        artifacts: list[dict[str, Any]] = []
        for name in sorted(selected_names):
            path = run / name
            if not path.is_file() or path.is_symlink() or not path.stat().st_size:
                continue
            archive_name = relative_run / name
            artifacts.append(
                {
                    "path": archive_name.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
            files_to_archive.append((path, archive_name))
        runs.append(
            {
                "relative_path": relative_run.as_posix() or ".",
                "artifacts": artifacts,
            }
        )

    manifest = {
        "format": "interfaceforge-vasp-mlff-model-archive",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "selection": "directories containing a nonempty ML_AB; scientific acceptance is user-supplied",
        "include_large": include_large,
        "potcar_excluded": True,
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path, archive_name in files_to_archive:
                archive.write(path, archive_name.as_posix())
            archive.writestr(
                _MODEL_ARCHIVE_MANIFEST,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "root": str(source_root),
        "output": str(output_path),
        "archive_sha256": _sha256_file(output_path),
        "runs": len(runs),
        "files": len(files_to_archive),
        "include_large": include_large,
        "manifest": _MODEL_ARCHIVE_MANIFEST,
        "potcar_excluded": True,
    }
