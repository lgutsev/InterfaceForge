"""Shared fixtures for the density-initialization tests (no ML stack needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# NiO AFM-II-like cell: Ni1 up, Ni2 down, two O.  The signed MAGMOM is the
# authoritative magnetic branch the initializer must never replace.
NIO_POSCAR = (
    "NiO AFM-II\n1.0\n4.17 2.085 2.085\n2.085 4.17 2.085\n2.085 2.085 4.17\n"
    "Ni O\n2 2\nDirect\n0 0 0\n0.5 0.5 0.5\n0.25 0.25 0.25\n0.75 0.75 0.75\n"
)
NIO_MAGMOM = "1.7 -1.7 2*0"
NIO_INCAR = (
    "SYSTEM = NiO AFM-II   ! signed moments are authoritative\n"
    "ENCUT = 520\nPREC = Accurate\nEDIFF = 1E-6\nISPIN = 2\n"
    f"MAGMOM = {NIO_MAGMOM}\n"
    "LDAU = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2 -1\nLDAUU = 6.2 0.0\nLDAUJ = 0.0 0.0\n"
    "LMAXMIX = 4\nLORBIT = 11\nISTART = 0\n"
)
STO_POSCAR = (
    "SrTiO3\n1.0\n3.905 0 0\n0 3.905 0\n0 0 3.905\nSr Ti O\n1 1 3\nDirect\n"
    "0 0 0\n0.5 0.5 0.5\n0.5 0.5 0\n0.5 0 0.5\n0 0.5 0.5\n"
)
STO_INCAR = "ENCUT = 520\nPREC = Accurate\nEDIFF = 1E-6\nISMEAR = 0\nSIGMA = 0.05\n"
KPOINTS = "auto\n0\nGamma\n4 4 4\n0 0 0\n"

_DATASETS = {
    # symbol: (titel, zval, [(l, nproj), ...]) -- l-channels match the MP set
    "Ni_pv": ("PAW_PBE Ni_pv 06Sep2000", 16.0, [(1, 2), (2, 2), (0, 2)]),
    "Ni": ("PAW_PBE Ni 02Aug2007", 10.0, [(2, 2), (0, 2)]),
    "O": ("PAW_PBE O 08Apr2002", 6.0, [(0, 2), (1, 2)]),
    "Sr_sv": ("PAW_PBE Sr_sv 07Sep2000", 10.0, [(0, 2), (1, 2), (2, 2)]),
    "Ti_pv": ("PAW_PBE Ti_pv 07Sep2000", 10.0, [(1, 2), (2, 2), (0, 2)]),
}


def potcar(*symbols: str) -> str:
    blocks = []
    for symbol in symbols:
        titel, zval, channels = _DATASETS[symbol]
        lines = [
            f"  {titel}",
            f"   TITEL  = {titel}",
            f"   POMASS =   1.000; ZVAL   = {zval:8.3f}    mass and valenz",
        ]
        for l_value, count in channels:
            lines += [" Non local Part", f"    {l_value}    {count}    .12345678E+01"]
        lines.append(" End of Dataset")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def write_run(
    run: Path,
    *,
    poscar: str = NIO_POSCAR,
    incar: str = NIO_INCAR,
    potcar_symbols: tuple[str, ...] = ("Ni_pv", "O"),
    kpoints: bool = True,
) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    (run / "POSCAR").write_text(poscar, encoding="utf-8")
    (run / "INCAR").write_text(incar, encoding="utf-8")
    (run / "POTCAR").write_text(potcar(*potcar_symbols), encoding="utf-8")
    if kpoints:
        (run / "KPOINTS").write_text(KPOINTS, encoding="utf-8")
    return run


class FakeRunner:
    """Stands in for the neural_paw_dft worker subprocess.

    It honours the request/result protocol exactly as the real worker does,
    writing a CHGCAR whose header is the POSCAR copy and whose grid line is
    the requested grid.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        fail: str | None = None,
        tamper: Path | None = None,
        moments: list[float] | None = None,
        wrong_grid: bool = False,
    ) -> None:
        self.available = available
        self.fail = fail
        self.tamper = tamper
        self.moments = moments
        self.wrong_grid = wrong_grid
        self.requests: list[dict[str, Any]] = []
        self.calls: list[list[str]] = []

    @property
    def inference_calls(self) -> int:
        return len(self.requests)

    def __call__(self, argv: list[str], cwd: Path, log: Path, timeout: float | None) -> int:
        self.calls.append(argv)
        log.write_text("fake neural_paw_dft worker\n", encoding="utf-8")
        if argv[2] == "--probe":
            payload = (
                {"available": True, "version": "0.1.0", "commit": "abc1234", "detail": "fake"}
                if self.available
                else {"available": False, "detail": "ModuleNotFoundError: No module named 'neural_paw_dft'"}
            )
            Path(argv[3]).write_text(json.dumps(payload), encoding="utf-8")
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        self.requests.append(request)
        if self.tamper is not None:
            self.tamper.write_text(self.tamper.read_text() + "NELM = 999\n", encoding="utf-8")
        if self.fail:
            Path(request["result"]).write_text(
                json.dumps({"status": "error", "error": self.fail, "timing": {}}), encoding="utf-8"
            )
            return 1
        grid = [g + 1 for g in request["grid"]] if self.wrong_grid else request["grid"]
        poscar = Path(request["poscar"]).read_text(encoding="utf-8")
        Path(request["chgcar"]).write_text(
            poscar.rstrip("\n") + "\n\n" + " ".join(f"{g:5d}" for g in grid) + "\n 0.1 0.2 0.3 0.4 0.5\n",
            encoding="utf-8",
        )
        n_ions = len(request["species"])
        moments = None
        if request["use_initializer_moments"]:
            moments = self.moments or [0.9] * n_ions
        Path(request["result"]).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "package": "neural_paw_dft",
                    "version": "0.1.0",
                    "commit": "abc1234",
                    "python": "3.11.0",
                    "torch": "2.4.1",
                    "device": "cpu",
                    "models": {
                        "electrafi": {
                            "name": "electrafi_spin_constrained" if request["spin_channel"] else "electrafi_total",
                            "sha256": "e" * 64,
                        },
                        "augnet_total": {"name": "augnet_total_full", "sha256": "a" * 64},
                    },
                    "spin_channel_written": request["spin_channel"],
                    "initializer_moments": moments,
                    "potcar_warnings": [],
                    "total_integral": request["nelect"],
                    "timing": {"model_load_s": 2.0, "inference_s": 1.25, "write_s": 0.1},
                }
            ),
            encoding="utf-8",
        )
        return 0
