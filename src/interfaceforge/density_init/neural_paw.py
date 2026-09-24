"""``neural-paw`` backend: the neural_paw_dft Complete Neural Electronic Initializer.

Upstream: https://github.com/aerte/neural_paw_dft (arXiv:2609.21759).  The
package is *not* an InterfaceForge dependency: it pins its own
``torch``/``mace-torch==0.3.16``/``e3nn``/``pykeops`` stack (which conflicts
with ``interfaceforge[mace-roi]``), is CC-BY-NC-4.0 licensed, and downloads
weights from Hugging Face on first use.  InterfaceForge therefore drives it
through a small stdlib-only worker script executed by a configurable Python
interpreter (``--ndi-python`` / ``$IFACE_NDI_PYTHON``, default: the current
interpreter), in a private scratch directory holding *copies* of the inputs.
A crash, OOM or import error in the ML stack can therefore never touch the
run directory, and the orchestration process never imports torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import DependencyError, InterfaceForgeError
from .base import BackendResult, DensityInitializer, InitializationPlan
from .inputs import potcar_entries

WORKER = Path(__file__).with_name("neural_paw_worker.py")

INSTALL_HINT = (
    "Install neural_paw_dft in its own environment, e.g.\n"
    "  python -m venv ~/envs/ndi && ~/envs/ndi/bin/pip install 'torch>=2.4.1' "
    "'git+https://github.com/aerte/neural_paw_dft'\n"
    "then pass --ndi-python ~/envs/ndi/bin/python (or export IFACE_NDI_PYTHON). "
    "See docs/density-init.md."
)

#: ``runner(argv, cwd, log, timeout) -> returncode``.  Injected by tests.
Runner = Callable[[list[str], Path, Path, float | None], int]


class InferenceError(InterfaceForgeError):
    """The initializer ran but did not produce a usable initial state.

    ``code`` is a machine-readable reason when one is known (e.g.
    ``UNSUPPORTED_POTCAR_SCHEMA``); it is recorded as ``failure_code``.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def subprocess_runner(argv: list[str], cwd: Path, log: Path, timeout: float | None) -> int:
    env = dict(os.environ)
    # e3nn 0.4 + torch>=2.6: the upstream package sets this itself, but only
    # after torch may already have been imported by a sitecustomize.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with log.open("ab") as handle:
        try:
            completed = subprocess.run(
                argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            handle.write(f"\nInterfaceForge: worker exceeded timeout of {timeout} s\n".encode())
            return 124
        except OSError as exc:
            handle.write(f"\nInterfaceForge: cannot execute worker: {exc}\n".encode())
            return 127
    return completed.returncode


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])
    except OSError:
        return ""


class NeuralPawInitializer(DensityInitializer):
    name = "neural-paw"
    writes_chgcar = True
    supports_spin_channel = True
    has_automatic_moments = True  # CHGNet (unsigned) site moments
    needs_grid = True

    def __init__(
        self,
        *,
        python: str | None = None,
        device: str = "auto",
        weights_dir: str | None = None,
        config: str | None = None,
        electrafi_checkpoint: str | None = None,
        augnet_total_checkpoint: str | None = None,
        augnet_mag_checkpoint: str | None = None,
        allow_potcar_variant: bool = False,
        timeout: float | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.python = python or os.environ.get("IFACE_NDI_PYTHON") or sys.executable
        self.device = device
        self.weights_dir = weights_dir
        self.config = str(Path(config).expanduser().resolve()) if config else None
        self.electrafi_checkpoint = electrafi_checkpoint
        self.augnet_total_checkpoint = augnet_total_checkpoint
        self.augnet_mag_checkpoint = augnet_mag_checkpoint
        self.allow_potcar_variant = allow_potcar_variant
        self.timeout = timeout
        self.runner = runner or subprocess_runner

    # ------------------------------------------------------------------
    def settings(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "device": self.device,
            "weights_dir": self.weights_dir,
            "config": self.config,
            "electrafi_checkpoint": self.electrafi_checkpoint,
            "augnet_total_checkpoint": self.augnet_total_checkpoint,
            "augnet_mag_checkpoint": self.augnet_mag_checkpoint,
            "allow_potcar_variant": self.allow_potcar_variant,
            "timeout_s": self.timeout,
        }

    def _call(self, args: list[str], workdir: Path, out: Path) -> tuple[int, dict[str, Any] | None, Path]:
        workdir.mkdir(parents=True, exist_ok=True)
        log = workdir / "worker.log"
        code = self.runner([self.python, str(WORKER), *args], workdir, log, self.timeout)
        return code, _read_json(out), log

    def probe(self, workdir: Path | None = None) -> dict[str, Any]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="iface_ndi_probe_") as tmp:
            scratch = Path(workdir or tmp)
            out = scratch / "probe.json"
            code, payload, log = self._call(["--probe", str(out)], scratch, out)
            if payload is None:
                return {
                    "available": False,
                    "python": self.python,
                    "detail": f"probe exited {code} without output: {_tail(log, 5)}",
                }
            return {"python": self.python, **payload}

    def require_available(self) -> dict[str, Any]:
        info = self.probe()
        if not info.get("available"):
            raise DependencyError(
                f"neural-paw backend unavailable via {self.python}: {info.get('detail')}\n{INSTALL_HINT}"
            )
        return info

    def prefetch(self, *, spin: bool = True) -> dict[str, Any]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="iface_ndi_prefetch_") as tmp:
            scratch = Path(tmp)
            out = scratch / "prefetch.json"
            args = ["--prefetch", str(out)]
            if self.weights_dir:
                args += ["--weights-dir", self.weights_dir]
            if not spin:
                args.append("--no-spin")
            code, payload, log = self._call(args, scratch, out)
            if code != 0 or not payload or payload.get("status") != "ok":
                raise DependencyError(
                    f"weight prefetch failed ({(payload or {}).get('error') or _tail(log, 5)})\n{INSTALL_HINT}"
                )
            return payload

    # ------------------------------------------------------------------
    def generate(self, plan: InitializationPlan, workdir: Path) -> BackendResult:
        if plan.grid is None:
            raise InferenceError("neural-paw needs the target FFT grid (NGXF NGYF NGZF)")
        magnetism = plan.magnetism
        request_path = workdir / "request.json"
        result_path = workdir / "result.json"
        chgcar = workdir / "CHGCAR"
        potcar = potcar_entries(workdir / "POTCAR")
        request = {
            "poscar": str(workdir / "POSCAR"),
            "potcar": [
                {"symbol": entry["symbol"], "titel": entry["titel"], "projector_l": entry["projector_l"]}
                for entry in potcar
            ],
            "species": plan.species,
            "grid": list(plan.grid),
            "nelect": plan.nelect,
            "lmaxmix": plan.lmaxmix,
            "spin_channel": bool(magnetism["spin_channel"]),
            "use_initializer_moments": bool(magnetism["use_initializer_moments"]),
            "site_moments": magnetism["site_moments_passed"],
            "device": self.device,
            "weights_dir": self.weights_dir,
            "config": self.config,
            "electrafi_checkpoint": self.electrafi_checkpoint,
            "augnet_total_checkpoint": self.augnet_total_checkpoint,
            "augnet_mag_checkpoint": self.augnet_mag_checkpoint,
            "allow_potcar_variant": self.allow_potcar_variant,
            "chgcar": str(chgcar),
            "result": str(result_path),
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        started = time.perf_counter()
        code, payload, log = self._call([str(request_path)], workdir, result_path)
        elapsed = time.perf_counter() - started
        if payload is None:
            raise InferenceError(f"neural-paw worker exited {code} without a result:\n{_tail(log)}")
        if code != 0 or payload.get("status") != "ok":
            raise InferenceError(
                f"neural-paw inference failed (exit {code}): {payload.get('error')}\n{_tail(log, 10)}",
                code=payload.get("error_code"),
            )
        if not (chgcar.is_file() and chgcar.stat().st_size):
            raise InferenceError("neural-paw worker reported success but wrote no CHGCAR")
        if bool(payload.get("spin_channel_written")) != bool(magnetism["spin_channel"]):
            raise InferenceError(
                "neural-paw spin channel does not match the magnetism plan "
                f"(planned {magnetism['spin_channel']}, written {payload.get('spin_channel_written')})"
            )
        timing = {key: float(value) for key, value in (payload.get("timing") or {}).items()}
        timing["subprocess_s"] = elapsed
        warnings = [f"POTCAR: {message}" for message in payload.get("potcar_warnings") or []]
        return BackendResult(
            chgcar=chgcar,
            backend_info={
                "backend": self.name,
                "package": payload.get("package", "neural_paw_dft"),
                "package_version": payload.get("version"),
                "package_commit": payload.get("commit"),
                "package_source_url": payload.get("source_url"),
                "python": payload.get("python"),
                "interpreter": self.python,
                "torch": payload.get("torch"),
                "device": payload.get("device"),
                "models": payload.get("models"),
                "settings": self.settings(),
            },
            timing=timing,
            warnings=warnings,
            initializer_moments=payload.get("initializer_moments"),
            spin_channel_written=bool(payload.get("spin_channel_written")),
            log=log,
            extra={
                key: payload.get(key)
                for key in ("total_integral", "spin_integral", "m_total_constraint", "grid", "nelect", "lmaxmix")
            },
        )
