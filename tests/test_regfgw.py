from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.regfgw import compare_registry_selection, regfgw_status


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RegfgwStatusTests(unittest.TestCase):
    def test_status_reports_unverified_output_schema(self) -> None:
        payload = regfgw_status()

        self.assertFalse(payload["output_schema_verified"])
        self.assertIn("install", payload)
        self.assertEqual(payload["adapter"], "interfaceforge.regfgw")


class CompareRegistrySelectionTests(unittest.TestCase):
    def _exhaustive_grid(self, root: Path) -> Path:
        # reg_3 is the true best (highest work of adhesion = most stable).
        path = root / "exhaustive.csv"
        _write_csv(
            path,
            [
                {"registry_id": "reg_1", "work_of_adhesion_ev_a2": 0.30},
                {"registry_id": "reg_2", "work_of_adhesion_ev_a2": 0.55},
                {"registry_id": "reg_3", "work_of_adhesion_ev_a2": 0.80},
                {"registry_id": "reg_4", "work_of_adhesion_ev_a2": 0.20},
                {"registry_id": "reg_5", "work_of_adhesion_ev_a2": 0.60},
            ],
            ["registry_id", "work_of_adhesion_ev_a2"],
        )
        return path

    def test_top_k_that_finds_the_true_best_has_zero_regret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = self._exhaustive_grid(root)
            topk = root / "topk.csv"
            _write_csv(
                topk,
                [{"registry_id": r} for r in ("reg_3", "reg_5", "reg_2")],
                ["registry_id"],
            )

            result = compare_registry_selection(topk, exhaustive, root / "result.csv")

            by_k = {row["k"]: row for row in result["results"]}
            self.assertEqual(by_k[1]["recall_at_k"], 1.0)
            self.assertTrue(by_k[1]["best_preserved"])
            self.assertEqual(by_k[1]["energy_regret"], 0.0)
            self.assertEqual(result["true_best_id"], "reg_3")

    def test_top_k_that_misses_the_true_best_has_nonzero_regret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = self._exhaustive_grid(root)
            topk = root / "topk.csv"
            _write_csv(
                topk,
                [{"registry_id": r} for r in ("reg_1", "reg_4", "reg_2")],
                ["registry_id"],
            )

            result = compare_registry_selection(topk, exhaustive, root / "result.csv")

            by_k = {row["k"]: row for row in result["results"]}
            self.assertEqual(by_k[1]["recall_at_k"], 0.0)
            self.assertFalse(by_k[1]["best_preserved"])
            self.assertAlmostEqual(by_k[1]["energy_regret"], 0.5)
            self.assertTrue(Path(result["output"]).is_file())
            self.assertTrue(Path(result["output"]).with_suffix(".json").is_file())

    def test_lower_energy_is_better_flips_the_ranking_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = root / "exhaustive.csv"
            _write_csv(
                exhaustive,
                [
                    {"registry_id": "reg_1", "formation_energy_ev": -2.0},
                    {"registry_id": "reg_2", "formation_energy_ev": -0.5},
                ],
                ["registry_id", "formation_energy_ev"],
            )
            topk = root / "topk.csv"
            _write_csv(topk, [{"registry_id": "reg_1"}], ["registry_id"])

            result = compare_registry_selection(
                topk,
                exhaustive,
                root / "result.csv",
                energy_column="formation_energy_ev",
                lower_energy_is_better=True,
                k_values=(1,),
            )

            self.assertEqual(result["true_best_id"], "reg_1")
            self.assertTrue(result["results"][0]["best_preserved"])

    def test_proposed_id_missing_from_exhaustive_grid_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = self._exhaustive_grid(root)
            topk = root / "topk.csv"
            _write_csv(topk, [{"registry_id": "reg_unknown"}], ["registry_id"])

            result = compare_registry_selection(
                topk, exhaustive, root / "result.csv", k_values=(1,)
            )

            row = result["results"][0]
            self.assertEqual(row["proposed_ids_missing_from_grid"], 1)
            self.assertIsNone(row["energy_regret"])

    def test_requires_id_column_in_topk_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = self._exhaustive_grid(root)
            topk = root / "topk.csv"
            _write_csv(topk, [{"not_id": "reg_1"}], ["not_id"])

            with self.assertRaises(ValueError):
                compare_registry_selection(topk, exhaustive, root / "result.csv")

    def test_output_json_matches_csv_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exhaustive = self._exhaustive_grid(root)
            topk = root / "topk.csv"
            _write_csv(topk, [{"registry_id": "reg_3"}], ["registry_id"])

            result = compare_registry_selection(
                topk, exhaustive, root / "result.csv", k_values=(1, 3)
            )
            payload = json.loads(Path(result["output"]).with_suffix(".json").read_text())

            self.assertEqual(len(payload["results"]), 2)


if __name__ == "__main__":
    unittest.main()
