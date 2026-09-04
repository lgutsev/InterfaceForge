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
    _model_member_labels,
    merge_separation_energy,
    separation_energy,
    write_json_payload,
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

    def test_same_named_deepmd_models_get_unique_member_labels(self) -> None:
        labels = _model_member_labels(
            [
                "models/deepmd/dpa2/model_000/frozen_model.pth",
                "models/deepmd/dpa2/model_001/frozen_model.pth",
                "models/deepmd/dpa2/model_002/frozen_model.pth",
            ]
        )
        self.assertEqual(
            labels,
            [
                "model_000/frozen_model",
                "model_001/frozen_model",
                "model_002/frozen_model",
            ],
        )

    def test_duplicate_model_path_is_rejected(self) -> None:
        with self.assertRaises(SafetyError):
            _model_member_labels(["model.pth", "model.pth"])


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

    def test_backend_isolated_payloads_merge_into_one_report(self) -> None:
        def fake_mace(models, atoms_by_part, device):
            return {
                "seed_11": {"interface": -200.0, "slab_a": -95.0, "slab_b": -97.0},
                "seed_23": {"interface": -200.2, "slab_a": -95.0, "slab_b": -97.0},
            }

        def fake_deepmd(models, atoms_by_part):
            return {
                "model_000/frozen_model": {
                    "interface": -199.0,
                    "slab_a": -95.0,
                    "slab_b": -97.0,
                },
                "model_001/frozen_model": {
                    "interface": -199.2,
                    "slab_a": -95.0,
                    "slab_b": -97.0,
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n_set = _set(root, "nterm", -200.0, -95.0, -97.0)
            with patch("interfaceforge.separation_energy._mace_energies", fake_mace):
                mace = separation_energy(
                    [("nterm-111", n_set)],
                    mace_models=["seed_11.model", "seed_23.model"],
                    campaign_validation=_VALIDATION,
                )
            with patch("interfaceforge.separation_energy._deepmd_energies", fake_deepmd):
                deepmd = separation_energy(
                    [("nterm-111", n_set)],
                    deepmd_models=["model_000.pth", "model_001.pth"],
                    campaign_validation=_VALIDATION,
                )
            mace_path = root / "mace.json"
            deepmd_path = root / "deepmd.json"
            mace_path.write_text(json.dumps(mace), encoding="utf-8")
            deepmd_path.write_text(json.dumps(deepmd), encoding="utf-8")

            merged = merge_separation_energy(
                [mace_path, deepmd_path], campaign_validation=_VALIDATION
            )
            row = merged["interfaces"][0]
            self.assertEqual(set(row["mlip"]), {"mace", "deepmd"})
            self.assertEqual(row["mlip"]["mace"]["members"], 2)
            self.assertEqual(row["mlip"]["deepmd"]["members"], 2)
            self.assertIn("delta_vs_dft_j_per_m2", row["mlip"]["mace"])
            self.assertEqual({hit["source"] for hit in row["literature"]}, {"dft", "mace", "deepmd"})
            self.assertEqual(merged["merged_from"], [str(mace_path), str(deepmd_path)])

            outputs = write_reports(merged, root / "merged")
            self.assertTrue(Path(outputs["csv"]).is_file())
            rows = list(csv.DictReader(Path(outputs["csv"]).open()))
            self.assertEqual({item["source"] for item in rows}, {"dft", "mace", "deepmd"})

    def test_merge_rejects_different_interface_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = separation_energy([("nterm-111", _set(root, "nterm", -200.0, -95.0, -97.0))])
            second = separation_energy([("titerm-111", _set(root, "titerm", -210.0, -95.0, -97.0))])
            first_path = root / "first.json"
            second_path = root / "second.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second_path.write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "interface specs differ"):
                merge_separation_energy([first_path, second_path])

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

    def test_json_only_partial_does_not_render_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = separation_energy(
                [("nterm-111", _set(root, "nterm", -200.0, -95.0, -97.0))]
            )
            outputs = write_json_payload(payload, root / "partial")
            self.assertEqual(set(outputs), {"json"})
            self.assertTrue(Path(outputs["json"]).is_file())
            self.assertFalse((root / "partial" / "separation_energy.csv").exists())
            self.assertFalse((root / "partial" / "separation_energy.png").exists())

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

    def test_reads_an_adhesion_prepare_tree_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "N_term_adhesion_dft"
            _part(tree / "interface_static", "Si Ti N", 12, -200.0)
            _part(tree / "slabs" / "lower", "Si N", 6, -95.0)
            _part(tree / "slabs" / "upper", "Ti N", 6, -97.0)
            (tree / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "dft",
                        "slab_mode": "static",
                        "reference_directory": str(tree / "interface_static"),
                        "interface_static": {"directory": "interface_static"},
                        "slabs": [
                            {"name": "lower", "directory": "slabs/lower"},
                            {"name": "upper", "directory": "slabs/upper"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = separation_energy(
                [("interface/450K/Real/N_Term/SiN_TiN_N-term", tree)],
                campaign_validation=_VALIDATION,
            )
            self.assertEqual(payload["slab_modes"], ["static"])
            self.assertIn("work of separation", payload["definition"])
            row = payload["interfaces"][0]
            self.assertEqual(row["slab_mode"], "static")
            self.assertAlmostEqual(row["dft"]["gamma_sep_j_per_m2"], 8.0 / 132.0 * 16.02176634)


if __name__ == "__main__":
    unittest.main()
