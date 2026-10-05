from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from interfaceforge.rescue_intake import (
    electronic_trace,
    geometry_difference,
    label_difference,
    scf_summary,
    verify_resettle,
)

SETTLED_THEN_JUMP = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
RMM:   1    -0.100000000000E+02   -0.10000E+02   -0.10000E+02   100   0.100E+01
RMM:   2    -0.200000000000E+02   -0.10000E+02   -0.50000E+01   100   0.500E+00    0.200E+00
RMM:   3    -0.200005000000E+02   -0.50000E-03   -0.40000E-03   100   0.100E-01    0.100E-02
RMM:   4    -0.200005100000E+02   -0.10000E-04   -0.10000E-04   100   0.100E-02    0.400E-04
RMM:   5    -0.172000000000E+02    0.28000E+01    0.10000E+00   100   0.100E+00    0.460E-01
RMM:   6    -0.172000000100E+02   -0.10000E-07   -0.90000E-06   100   0.100E-03
   1 F= -.17200000E+02 E0= -.17190000E+02  d E =-.172000E+02
"""

CLEAN = """\
DAV:   1    -0.100000000000E+02   -0.10000E+02   -0.10000E+02   100   0.100E+01
DAV:   2    -0.200000000000E+02   -0.10000E+02   -0.50000E+01   100   0.500E+00    0.200E+00
DAV:   3    -0.200005000000E+02   -0.50000E-03   -0.40000E-03   100   0.100E-01    0.100E-02
DAV:   4    -0.200005100000E+02   -0.10000E-07   -0.10000E-07   100   0.100E-02
   1 F= -.20000510E+02 E0= -.20000500E+02  d E =-.200005E+02
"""


def _run(tmp_path: Path, oszicar: str) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "OSZICAR").write_text(oszicar)
    return run


def test_energy_rise_after_settling_isolates_the_frame_0009_pattern(tmp_path):
    summary = scf_summary(_run(tmp_path, SETTLED_THEN_JUMP), {"EDIFF": "1E-6", "NELM": "150", "ALGO": "Fast"})
    assert summary["settled_iteration"] == 3
    assert summary["max_rise_after_settled_ev"] == pytest.approx(2.80051, abs=1e-5)
    assert summary["residual_jumps"] == 1
    assert summary["energy_criterion_met"] is True  # the energy test alone passes
    assert summary["final_step_rms_c"] is None and summary["last_rms_c"] == pytest.approx(0.046)


def test_clean_scf_has_no_rise_and_defaults_are_vasp_defaults(tmp_path):
    summary = scf_summary(_run(tmp_path, CLEAN), {})
    assert summary["ediff_ev"] == 1e-4 and summary["nelm"] == 60 and summary["algo"] == "Normal"
    assert summary["max_rise_after_settled_ev"] <= 0
    assert summary["residual_jumps"] == 0
    assert len(electronic_trace(tmp_path / "run" / "OSZICAR")) == 4


def _label(energy: float, forces: np.ndarray, stress: np.ndarray, positions: np.ndarray | None = None):
    cell = np.diag([5.0, 5.0, 5.0])
    pos = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]) if positions is None else positions
    return {
        "symbols": ["Ti", "N"],
        "positions": pos,
        "cell": cell,
        "energy_sigma0": energy,
        "free_energy": energy - 0.1,
        "forces": forces,
        "stress_gpa": stress,
    }


def test_label_difference_reports_all_channels():
    ref = _label(-10.0, np.zeros((2, 3)), np.zeros((3, 3)))
    test = _label(-9.9, np.array([[0.3, 0.0, 0.0], [0.0, -0.4, 0.0]]), np.diag([1.0, -2.0, 0.5]))
    diff = label_difference(test, ref)
    assert diff["dE_sigma0_mev_atom"] == pytest.approx(50.0)
    assert diff["dF_free_ev"] == pytest.approx(0.1)
    assert diff["force_rmse_ev_a"] == pytest.approx(np.sqrt((0.09 + 0.16) / 6))
    assert diff["force_max_component_ev_a"] == pytest.approx(0.4)
    assert diff["stress_max_abs_diff_gpa"] == pytest.approx(2.0)
    assert diff["same_species_order"] is True


def test_geometry_difference_uses_minimum_image():
    a = _label(0, np.zeros((2, 3)), np.zeros((3, 3)), np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]))
    b = _label(0, np.zeros((2, 3)), np.zeros((3, 3)), np.array([[5.0, 0.0, 0.0], [2.5, 2.5, 2.5 + 1e-3]]))
    diff = geometry_difference(a, b)
    assert diff["max_position_diff_a"] == pytest.approx(1e-3)
    b["symbols"] = ["N", "Ti"]
    assert geometry_difference(a, b)["same_species_order"] is False


def test_verify_resettle_lists_missing_and_mismatched_members(tmp_path):
    package = tmp_path / "pkg"
    (package / "runs").mkdir(parents=True)
    good = package / "README.md"
    good.write_text("readme")
    bad = package / "runs" / "OUTCAR"
    bad.write_text("changed")
    rows = [
        (hashlib.sha256(b"readme").hexdigest(), 6, "README.md"),
        (hashlib.sha256(b"original").hexdigest(), 8, "runs/OUTCAR"),
        (hashlib.sha256(b"gone").hexdigest(), 4, "runs/OSZICAR"),
    ]
    lines = ["# resettle audit", "# part", "# left out: WAVECAR", "sha256\tbytes\tpath\ttarget"]
    lines += [f"{s}\t{n}\t{p}\t" for s, n, p in rows]
    (package / "RESETTLE_AUDIT.tsv").write_text("\n".join(lines) + "\n")
    (package / "extra.txt").write_text("x")
    report = verify_resettle(package)
    assert report["verified"] == 1
    assert report["mismatched"] == ["runs/OUTCAR"]
    assert report["missing"] == ["runs/OSZICAR"]
    assert report["unlisted_local"] == ["extra.txt"]
    assert report["complete"] is False
