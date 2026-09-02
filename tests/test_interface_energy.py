from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.errors import SafetyError
from interfaceforge.interface_energy import interface_energy, write_reports


def _system(
    root: Path,
    split: str,
    leaf: str,
    symbols: list[str],
    cell: np.ndarray,
    energies: list[float],
    source_frames: list[int],
) -> None:
    system = root / split / leaf
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    type_map: list[str] = []
    for symbol in symbols:
        if symbol not in type_map:
            type_map.append(symbol)
    (system / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    (system / "type.raw").write_text(
        "\n".join(str(type_map.index(s)) for s in symbols) + "\n", encoding="utf-8"
    )
    nframes = len(energies)
    np.save(set_dir / "energy.npy", np.asarray(energies, dtype=float).reshape(nframes, 1))
    np.save(set_dir / "box.npy", np.tile(cell.reshape(1, 9), (nframes, 1)))
    np.save(set_dir / "coord.npy", np.zeros((nframes, len(symbols) * 3)))
    np.save(set_dir / "force.npy", np.zeros((nframes, len(symbols) * 3)))
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_frame", "source_frame", "source_path", "relative_leaf"])
        for local, source in enumerate(source_frames):
            writer.writerow([local, source, "OUTCAR", leaf])


def _dataset(root: Path) -> Path:
    deepmd = root / "datasets" / "canonical" / "deepmd"
    cell10 = np.diag([10.0, 10.0, 20.0])  # longest axis c -> area = 100 A^2

    # bulk SiN: 2 f.u. per cell, E/cell so E/fu = -10 (drop the first two equilibration frames)
    _system(deepmd, "train", "bulk/SiN-Bulk_300K", ["Si", "N", "Si", "N"], cell10,
            [999.0, 999.0, -20.0, -20.0, -20.0, -20.0], [0, 1, 2, 3, 4, 5])
    # bulk TiN: E/fu = -12
    _system(deepmd, "train", "bulk/TiN-Bulk_300K", ["Ti", "N", "Ti", "N"], cell10,
            [999.0, 999.0, -24.0, -24.0, -24.0, -24.0], [0, 1, 2, 3, 4, 5])
    # interface: 4 SiN + 4 TiN f.u., N balances (8). E_int = -100
    symbols = ["Si"] * 4 + ["Ti"] * 4 + ["N"] * 8
    _system(deepmd, "train", "interface/300K/Ideal/N_Term/SiN_TiN_N-term", symbols, cell10,
            [999.0, 999.0, -100.0, -100.0, -100.0, -100.0], [0, 1, 2, 3, 4, 5])
    return root


class TestInterfaceEnergy(unittest.TestCase):
    def test_bulk_referenced_gamma_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            payload = interface_energy(root, equilibration_frames=2, n_interfaces=2, blocks=4)

            self.assertIn("bulk/SiN-Bulk_300K", payload["bulk_references"])
            sin = payload["bulk_references"]["bulk/SiN-Bulk_300K"]
            self.assertEqual(sin["formula_units_per_cell"], 2)
            self.assertAlmostEqual(sin["energy_per_fu_ev"], -10.0)
            self.assertEqual(sin["frames_used"], 4)

            self.assertEqual(len(payload["interfaces"]), 1)
            row = payload["interfaces"][0]
            self.assertEqual(row["status"], "OK")
            self.assertTrue(row["nitrogen_balanced"])
            self.assertAlmostEqual(row["tin_formula_units"], 4.0)
            self.assertAlmostEqual(row["sin_formula_units"], 4.0)
            self.assertEqual(row["stacking_axis"], "c")
            self.assertAlmostEqual(row["interface_area_ang2"], 100.0)
            # excess = -100 - 4*(-12) - 4*(-10) = -12 eV ; /(2*100) ; *16.02176634
            self.assertAlmostEqual(row["gamma_int_ev_per_ang2"], -0.06)
            self.assertAlmostEqual(row["gamma_int_j_per_m2"], -0.06 * 16.02176634)
            # perfectly flat energies -> zero error
            self.assertAlmostEqual(row["gamma_int_sem_j_per_m2"], 0.0)

    def test_missing_reference_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            # add a 450 K interface with no 450 K bulk references
            symbols = ["Si"] * 4 + ["Ti"] * 4 + ["N"] * 8
            _system(root / "datasets" / "canonical" / "deepmd", "train",
                    "interface/450K/Real/Ti_Term/SiN-TiN-Ti-term",
                    symbols, np.diag([10.0, 10.0, 20.0]),
                    [-100.0] * 4, [2, 3, 4, 5])
            payload = interface_energy(root, equilibration_frames=2)
            by_leaf = {r["leaf"]: r for r in payload["interfaces"]}
            self.assertIn("missing bulk reference", by_leaf["interface/450K/Real/Ti_Term/SiN-TiN-Ti-term"]["status"])
            self.assertEqual(by_leaf["interface/300K/Ideal/N_Term/SiN_TiN_N-term"]["status"], "OK")

    def test_oxidized_interfaces_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            symbols = ["Si"] * 4 + ["Ti"] * 4 + ["N"] * 8 + ["O"] * 2
            _system(root / "datasets" / "canonical" / "deepmd", "train",
                    "interface/300K/Ideal/N_Term/SiN_TiN_N-term_O_x0.25",
                    symbols, np.diag([10.0, 10.0, 20.0]), [-110.0] * 4, [2, 3, 4, 5])
            payload = interface_energy(root, equilibration_frames=2)
            self.assertNotIn(
                "interface/300K/Ideal/N_Term/SiN_TiN_N-term_O_x0.25",
                {r["leaf"] for r in payload["interfaces"]},
            )

    def test_reports_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            payload = interface_energy(root, equilibration_frames=2)
            out = Path(temporary) / "report"
            paths = write_reports(payload, out)
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            data = json.loads((out / "interface_energy.json").read_text())
            self.assertEqual(data["schema_version"], 1)
            csv_rows = list(csv.DictReader((out / "interface_energy.csv").open()))
            self.assertAlmostEqual(float(csv_rows[0]["gamma_int_j_per_m2"]), -0.06 * 16.02176634)
            self.assertIn("-0.961", (out / "interface_energy.md").read_text())

    def test_empty_dataset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SafetyError):
                interface_energy(Path(temporary))


if __name__ == "__main__":
    unittest.main()
