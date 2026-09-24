"""Standalone neural_paw_dft worker (runs in the *initializer's* Python).

This file is executed as a script by :mod:`interfaceforge.density_init.neural_paw`
with an interpreter that has ``neural_paw_dft`` installed -- typically a
separate virtual environment, because the upstream package pins its own
``torch``/``mace-torch``/``e3nn`` stack.  It therefore imports nothing from
InterfaceForge: only the standard library, ``numpy``, ``pymatgen`` and
``neural_paw_dft``.

Modes::

    python neural_paw_worker.py --probe  OUT.json
    python neural_paw_worker.py --prefetch OUT.json [--weights-dir DIR] [--no-spin]
    python neural_paw_worker.py REQUEST.json

The worker only ever reads the POSCAR/POTCAR *copies* named in the request and
writes the CHGCAR and ``result.json`` named there.  It deliberately calls
``Pipeline.predict`` + ``assemble.build_chgcar`` instead of ``Pipeline.build``,
because ``build`` rewrites INCAR/POTCAR/KPOINTS with pymatgen's MPStaticSet
(and CHGNet MAGMOM), which would displace the authoritative InterfaceForge
inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def _write(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _package_info() -> dict:
    info: dict = {"package": "neural_paw_dft", "python": sys.version.split()[0], "platform": platform.platform()}
    import neural_paw_dft

    info["version"] = getattr(neural_paw_dft, "__version__", None)
    info["location"] = str(Path(neural_paw_dft.__file__).resolve().parent)
    try:
        from importlib import metadata

        dist = metadata.distribution("neural_paw_dft")
        direct = dist.read_text("direct_url.json")
        if direct:
            payload = json.loads(direct)
            info["source_url"] = payload.get("url")
            info["commit"] = (payload.get("vcs_info") or {}).get("commit_id")
            info["editable"] = bool((payload.get("dir_info") or {}).get("editable"))
    except Exception:  # noqa: BLE001 - provenance is best effort
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable: {exc}"
    return info


def probe(out: str) -> int:
    try:
        import inspect

        info = _package_info()
        from neural_paw_dft.pipeline import Pipeline, assemble

        # The adapter targets the packaged pipeline API (neural_paw_dft main,
        # commit bde513b); older layouts such as tag v1-paper lack it.
        needed = {"grid_dims", "nelect", "lmaxmix", "site_moments"}
        missing = needed - set(inspect.signature(Pipeline.predict).parameters)
        if missing or not callable(getattr(assemble, "build_chgcar", None)):
            raise ImportError(
                "incompatible neural_paw_dft: Pipeline.predict lacks "
                f"{sorted(missing) or 'nothing'} or assemble.build_chgcar is missing"
            )
        _write(out, {"available": True, "detail": "neural_paw_dft pipeline API compatible", **info})
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        _write(out, {"available": False, "detail": f"{type(exc).__name__}: {exc}"})
    return 0


def prefetch(out: str, weights_dir: str | None, spin: bool) -> int:
    """Download registry weights now (e.g. on a login node with network)."""

    try:
        from neural_paw_dft.models import resolve_weights

        names = ["electrafi_total", "augnet_total_full"]
        if spin:
            names += ["electrafi_spin_constrained", "augnet_spin_full"]
        files = {}
        for name in names:
            path = resolve_weights(name, weights_dir)
            files[name] = {"path": str(path), "sha256": _sha256(path)}
        _write(out, {"status": "ok", "weights": files, **_package_info()})
        return 0
    except Exception as exc:  # noqa: BLE001
        _write(out, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1


class PotcarRefused(RuntimeError):
    """The run's POTCAR cannot (or may not) be seeded by the models; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _potcar_check(request: dict, structure) -> tuple[list[str], list[str]]:
    """Compare the run's POTCAR datasets with the Materials Project set the models learned.

    Returns ``(schema_errors, variant_warnings)``.  A projector (l-channel)
    mismatch changes the size/layout of the augmentation block VASP expects,
    so it is never overridable; a different dataset name with an identical
    projector set is only out-of-distribution and may be allowed explicitly.
    """

    from neural_paw_dft.augnet.mp_potcar_map import MP_POTCAR_BY_Z
    from pymatgen.core import Element

    schema_errors: list[str] = []
    variant_warnings: list[str] = []
    datasets = request.get("potcar") or []
    elements = list(dict.fromkeys(site.specie.symbol for site in structure))
    if len(datasets) != len(elements):
        return [f"POTCAR has {len(datasets)} datasets but POSCAR has {len(elements)} species"], []
    for element, dataset in zip(elements, datasets, strict=True):
        entry = MP_POTCAR_BY_Z.get(Element(element).Z)
        if entry is None:
            schema_errors.append(f"{element}: not covered by the neural_paw_dft models")
            continue
        expected = sorted(int(x) for x in entry.get("l_channels", ()))
        found = sorted(int(x) for x in dataset.get("projector_l") or [])
        if found and found != expected:
            schema_errors.append(
                f"{element}: POTCAR {dataset.get('symbol')!r} has projector l-channels {found}, "
                f"the models' {entry.get('variant')!r} has {expected} (augmentation schema differs)"
            )
        elif not found:
            variant_warnings.append(f"{element}: could not read projector l-channels from POTCAR")
        if entry.get("variant") != dataset.get("symbol"):
            variant_warnings.append(
                f"{element}: run uses POTCAR {dataset.get('symbol')!r}, models were trained "
                f"with {entry.get('variant')!r}"
            )
    return schema_errors, variant_warnings


def run(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    result_path = Path(request["result"])
    timing: dict = {}
    started = time.perf_counter()
    try:
        import numpy as np
        from neural_paw_dft.pipeline import Pipeline, assemble, load_config
        from pymatgen.core import Structure

        info = _package_info()
        structure = Structure.from_file(request["poscar"])  # site order preserved
        species = [site.specie.symbol for site in structure]
        if species != request["species"]:
            raise RuntimeError("POSCAR species order read by pymatgen differs from InterfaceForge's")

        schema_errors, problems = _potcar_check(request, structure)
        if schema_errors:
            raise PotcarRefused(
                "UNSUPPORTED_POTCAR_SCHEMA", "POTCAR incompatible with the models: " + "; ".join(schema_errors)
            )
        if problems and not request.get("allow_potcar_variant"):
            raise PotcarRefused(
                "POTCAR_VARIANT_NOT_ALLOWED",
                "POTCAR datasets differ from the models' training set: "
                + "; ".join(problems)
                + " (pass --allow-potcar-variant to proceed out of distribution)"
            )

        spin = bool(request["spin_channel"])
        use_chgnet = bool(request["use_initializer_moments"])
        cfg = load_config(request.get("config"))
        changes: dict = {"grid_dims": tuple(int(x) for x in request["grid"])}
        if request.get("device"):
            changes["device"] = request["device"]
        if request.get("weights_dir"):
            changes["weights_dir"] = request["weights_dir"]
        cfg = dataclasses.replace(cfg, **changes)
        electrafi = cfg.electrafi
        if request.get("electrafi_checkpoint"):
            electrafi = dataclasses.replace(electrafi, checkpoint=request["electrafi_checkpoint"])
        augnet = cfg.augnet
        if request.get("augnet_total_checkpoint"):
            augnet = dataclasses.replace(augnet, total_checkpoint=request["augnet_total_checkpoint"])
        if request.get("augnet_mag_checkpoint"):
            augnet = dataclasses.replace(augnet, mag_checkpoint=request["augnet_mag_checkpoint"])
        if not spin:
            # same swap as `ndi --no-spin`: a spin checkpoint would run its head only to be discarded
            checkpoint = electrafi.checkpoint
            if str(checkpoint).startswith("electrafi_spin"):
                checkpoint = "electrafi_total"
            electrafi = dataclasses.replace(electrafi, spin=False, checkpoint=checkpoint)
            augnet = dataclasses.replace(augnet, mag_checkpoint=None)
        chgnet = dataclasses.replace(cfg.chgnet, enabled=use_chgnet)
        cfg = dataclasses.replace(cfg, electrafi=electrafi, augnet=augnet, chgnet=chgnet)

        pipe = Pipeline(cfg)
        t0 = time.perf_counter()
        pipe.electrafi(len(structure))
        pipe.augnet("total")
        if spin and cfg.augnet.mag_checkpoint:
            pipe.augnet("mag")
        if use_chgnet:
            pipe.chgnet()
        timing["model_load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_chgnet:
            moments = pipe.site_moments(structure)
        elif request.get("site_moments") is not None:
            moments = np.asarray(request["site_moments"], dtype=float)
        else:
            moments = None
        pred = pipe.predict(
            structure,
            grid_dims=tuple(int(x) for x in request["grid"]),
            nelect=float(request["nelect"]),
            lmaxmix=int(request["lmaxmix"]),
            site_moments=moments,
        )
        timing["inference_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = Path(request["chgcar"])
        assemble.build_chgcar(
            structure,
            pred.rho_total,
            pred.rho_spin,
            pred.aug_total,
            pred.aug_diff,
            pred.site_moments,
            out,
            nelect=float(request["nelect"]),
        )
        timing["write_s"] = time.perf_counter() - t0
        timing["worker_total_s"] = time.perf_counter() - started

        from neural_paw_dft.models import REGISTRY, resolve_weights

        models = {}
        for role, name in (
            ("electrafi", cfg.electrafi.checkpoint),
            ("augnet_total", cfg.augnet.total_checkpoint),
            ("augnet_mag", cfg.augnet.mag_checkpoint if pred.aug_diff is not None else None),
        ):
            if not name:
                continue
            entry = {"name": name, "registry": name in REGISTRY}
            try:
                path = resolve_weights(name, cfg.weights_dir)
                entry.update(path=str(path), sha256=_sha256(path))
            except Exception as exc:  # noqa: BLE001
                entry["resolve_error"] = str(exc)
            models[role] = entry
        if use_chgnet:
            models["chgnet"] = {"name": cfg.chgnet.model or "CHGNet default"}

        volume = float(structure.volume)
        _write(
            result_path,
            {
                "status": "ok",
                **info,
                "device": str(pipe.device),
                "models": models,
                "grid": list(pred.grid_dims),
                "nelect": pred.nelect,
                "total_integral": float(pred.rho_total.sum() * volume / pred.rho_total.size),
                "spin_integral": None
                if pred.rho_spin is None
                else float(pred.rho_spin.sum() * volume / pred.rho_spin.size),
                "m_total_constraint": pred.m_total,
                "spin_channel_written": pred.rho_spin is not None,
                "initializer_moments": [float(x) for x in pred.site_moments]
                if (use_chgnet and pred.site_moments is not None)
                else None,
                "potcar_warnings": problems,
                "lmaxmix": pred.lmaxmix,
                "timing": timing,
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - reported to InterfaceForge, which fails safely
        timing["worker_total_s"] = time.perf_counter() - started
        _write(
            result_path,
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "error_code": getattr(exc, "code", None),
                "traceback": traceback.format_exc(),
                "timing": timing,
            },
        )
        return 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--probe":
        return probe(argv[1])
    if len(argv) >= 2 and argv[0] == "--prefetch":
        weights_dir = None
        if "--weights-dir" in argv:
            weights_dir = argv[argv.index("--weights-dir") + 1]
        return prefetch(argv[1], weights_dir, spin="--no-spin" not in argv)
    if len(argv) == 1:
        return run(argv[0])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
