from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interfaceforge.errors import SafetyError
from interfaceforge.separation_energy import (
    _family_block,
    _gamma,
    separation_energy,
    write_reports,
)

_POSCAR = """{title}
1.0
  12.0000000000  0.0000000000  0.0000000000
   0.0000000000 11.0000000000  0.0000000000
   0.0000000000  0.0000000000 28.0000000000
{symbols}
{counts}
Cartesian
{coords}
"""
_OUTCAR = (
    " energy  without entropy=     {without:.6f}  energy(sigma->0) =     {sigma0:.6f}\n"
    " General timing and accounting informations for this job\n"
)


def _part(directory: Path, symbols: str, count: int, energy: float | None) -> None:
    directory.mkdir(parents=True)
    coords = "\n".join(f"  {i * 0.5:.4f}  {i * 0.3:.4f}  {5 + i * 0.4:.4f}" for i in range(count))
    (directory / "POSCAR").write_text(
        _POSCAR.format(title=directory.name, symbols=symbols, counts=count, coords=coords),
        encoding="utf-8",
    )
    (directory / "INCAR").write_text("IBRION = -1\nNSW = 0\n", encoding="utf-8")
    if energy is not None:
        (directory / "OUTCAR").write_text(
            _OUTCAR.format(without=energy + 0.01, sigma0=energy), encoding="utf-8"
        )


def _set(root: Path, name: str, e_int: float | None, e_a: float | None, e_b: float | None) -> Path:
    base = root / name
    _part(base / "interface", "Si Ti N", 12, e_int)
    _part(base / "slab_a", "Si N", 6, e_a)
    _part(base / "slab_b", "Ti N", 6, e_b)
    return base


_VALIDATION = {
    "interfaces": [
        {"match": "*nterm*", "termination": "N"},
        {"match": "*titerm*", "termination": "Ti"},
    ],
    "references": [
        {
            "key": "sharifi2026",
            "quantity": "work_of_adhesion",
            "tolerance_j_per_m2": 0.5,
            "values": [
                {"match": {"termination": "N"}, "value_j_per_m2": 1.24},
                {"match": {"termination": "Ti"}, "value_j_per_m2": 3.28},
            ],
        }
    ],
}


class SeparationEnergyMathTests(unittest.TestCase):
    def test_gamma_is_slab_referenced_excess_over_area(self) -> None:
        gamma = _gamma({"interface": -200.0, "slab_a": -95.0, "slab_b": -97.0}, denom=132.0)
        self.assertAlmostEqual(gamma, (-95.0 - 97.0 + 200.0) / 132.0 * 16.02176634)

    def test_family_block_reports_ensemble_spread_and_delta(self) -> None:
        members = {
            "m0": {"interface": -200.0, "slab_a": -95.0, "slab_b": -97.0},
            "m1": {"interface": -201.0, "slab_a": -95.0, "slab_b": -97.0},
        }
        block = _family_block(members, denom=100.0, dft_gamma=1.0)
        self.assertEqual(block["members"], 2)
        self.assertGreater(block["committee_spread_j_per_m2"], 0.0)
        self.assertAlmostEqual(
            block["delta_vs_dft_j_per_m2"],
            block["gamma_sep_ensemble_j_per_m2"] - 1.0,
        )


class SeparationEnergyTests(unittest.TestCase):
    def test_dft_only_with_literature_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n_set = _set(root, "nterm", -200.0, -95.0, -97.0)
            ti_set = _set(root, "titerm", -210.0, -95.0, -97.0)
            payload = separation_energy(
                [("nterm-111", n_set), ("titerm-111", ti_set)],
                campaign_validation=_VALIDATION,
            )
            rows = payload["interfaces"]
            self.assertEqual(payload["quantity"], "separation_energy")
            self.assertEqual(rows[0]["area_axis"], "c")
            self.assertAlmostEqual(rows[0]["interface_area_ang2"], 132.0)
            # (-95 - 97 + 200) / 132 * conv
            self.assertAlmostEqual(rows[0]["dft"]["gamma_sep_j_per_m2"], 8.0 / 132.0 * 16.02176634)
            self.assertEqual(rows[0]["mlip"], {})
            lit = rows[0]["literature"]
            self.assertEqual(len(lit), 1)
            self.assertEqual(lit[0]["source"], "dft")
            self.assertEqual(lit[0]["reference_j_per_m2"], 1.24)

    def test_pending_dft_run_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unfinished = _set(root, "nterm", -200.0, None, -97.0)  # slab_a has no OUTCAR
            payload = separation_energy([("nterm-111", unfinished)])
            self.assertFalse(payload["interfaces"][0]["dft"]["ready"])
            self.assertIsNone(payload["interfaces"][0]["dft"]["gamma_sep_j_per_m2"])

    def test_mlip_committees_are_compared_against_dft(self) -> None:
        def fake_mace(models, atoms_by_part, device):
            return {
                f"mace_{k}": {"interface": -200.0 + 0.2 * k, "slab_a": -95.0, "slab_b": -97.0}
                for k in range(3)
            }

        def fake_deepmd(models, atoms_by_part):
            return {
                f"dpa_{k}": {"interface": -196.0 + 0.1 * k, "slab_a": -95.0, "slab_b": -97.0}
                for k in range(3)
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n_set = _set(root, "nterm", -200.0, -95.0, -97.0)
            with (
                patch("interfaceforge.separation_energy._mace_energies", fake_mace),
                patch("interfaceforge.separation_energy._deepmd_energies", fake_deepmd),
            ):
                payload = separation_energy(
                    [("nterm-111", n_set)],
                    mace_models=["a.model", "b.model", "c.model"],
                    deepmd_models=["x.pth"],
                )
            block = payload["interfaces"][0]["mlip"]
            self.assertEqual(set(block), {"mace", "deepmd"})
            self.assertEqual(block["mace"]["members"], 3)
            self.assertIn("delta_vs_dft_j_per_m2", block["mace"])
            # MACE ~ DFT here; DeePMD's interface is ~4 eV less bound -> smaller gamma_sep
            self.assertLess(abs(block["mace"]["delta_vs_dft_j_per_m2"]), 0.1)
            self.assertLess(block["deepmd"]["delta_vs_dft_j_per_m2"], -0.3)

    def test_reports_and_figure_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = separation_energy(
                [("nterm-111", _set(root, "nterm", -200.0, -95.0, -97.0))],
                campaign_validation=_VALIDATION,
            )
            outputs = write_reports(payload, root / "report")
            for key in ("json", "csv", "markdown"):
                self.assertTrue(Path(outputs[key]).is_file())
            data = json.loads((root / "report" / "separation_energy.json").read_text())
            self.assertEqual(data["reference"], "free-surface")
            rows = list(csv.DictReader((root / "report" / "separation_energy.csv").open()))
            self.assertEqual(rows[0]["source"], "dft")
            self.assertEqual(rows[0]["literature_key"], "sharifi2026")
            # matplotlib is a dev dependency, so the figure should render
            self.assertTrue(Path(outputs["figure_png"]).is_file())

    def test_bad_reference_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SafetyError):
                separation_energy(
                    [("x", _set(Path(temporary), "x", -1.0, -1.0, -1.0))], reference="vacuum"
                )

    def test_missing_subdirectory_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "incomplete"
            (root / "interface").mkdir(parents=True)
            with self.assertRaises(SafetyError):
                separation_energy([("x", root)])


if __name__ == "__main__":
    unittest.main()
