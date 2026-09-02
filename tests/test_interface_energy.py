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


def _add_test_split_and_predictions(root: Path) -> Path:
    """A 4-frame test split for the same leaves plus a fake mlip_compare tree."""
    deepmd = root / "datasets" / "canonical" / "deepmd"
    cell10 = np.diag([10.0, 10.0, 20.0])
    frames = [10, 11, 12, 13]
    specs = {
        "bulk/SiN-Bulk_300K": (["Si", "N", "Si", "N"], -20.0),
        "bulk/TiN-Bulk_300K": (["Ti", "N", "Ti", "N"], -24.0),
        "interface/300K/Ideal/N_Term/SiN_TiN_N-term": (["Si"] * 4 + ["Ti"] * 4 + ["N"] * 8, -100.0),
    }
    for leaf, (symbols, energy) in specs.items():
        _system(deepmd, "test", leaf, symbols, cell10, [energy] * 4, frames)

    out = root / "audit" / "mlip_compare"
    out.mkdir(parents=True)
    systems = [
        {"system_id": f"system_{i:03d}", "relative_leaf": leaf, "natoms": len(specs[leaf][0])}
        for i, leaf in enumerate(specs)
    ]
    models = [{"model": f"model_{i:03d}", "seed": s} for i, s in enumerate((11, 23))]
    (out / "comparison_manifest.json").write_text(
        json.dumps({"deepmd_architecture": "dpa2", "systems": systems, "models": models}),
        encoding="utf-8",
    )
    # model_000 predicts DFT exactly; model_001 shifts the interface by +4 eV
    for system in systems:
        for model in models:
            base = np.full(4, specs[system["relative_leaf"]][1], dtype=float)
            if model["model"] == "model_001" and system["relative_leaf"].startswith("interface/"):
                base = base + 4.0
            directory = out / "predictions" / "mace" / model["model"]
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{system['system_id']}.npz", energy=base, forces=np.zeros((4, 1, 3)))
    return out


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

    def test_mlip_mode_reports_committee_gamma_against_dft(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            predictions = _add_test_split_and_predictions(root)
            payload = interface_energy(
                root, predictions_root=predictions, equilibration_frames=2, blocks=4
            )
            row = payload["interfaces"][0]
            mlip = row["mlip"]
            self.assertEqual(mlip["status"], "OK")
            self.assertEqual(mlip["frames"], 4)
            # DFT on the test frames == the canonical DFT value here (flat energies)
            self.assertAlmostEqual(mlip["gamma_dft_same_frames_j_per_m2"], row["gamma_int_j_per_m2"], places=6)
            # model_000 reproduces DFT exactly; model_001 adds +4 eV to E_int
            members = mlip["gamma_members_j_per_m2"]
            self.assertAlmostEqual(members["model_000"], row["gamma_int_j_per_m2"], places=4)
            self.assertAlmostEqual(
                members["model_001"] - members["model_000"], 4.0 / (2 * 100.0) * 16.02176634, places=4
            )
            # ensemble = +2 eV on E_int -> delta = 2/(200) * conv
            self.assertAlmostEqual(mlip["delta_mlip_minus_dft_j_per_m2"], 2.0 / 200.0 * 16.02176634, places=4)
            self.assertGreater(mlip["member_spread_j_per_m2"], 0.0)

    def test_polar_termination_metadata_skips_the_leaf_with_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            metadata = [
                {
                    "match": "interface/*/*/N_Term/*",
                    "polar_termination": True,
                    "orientation": "Si3N4(0001)/TiN(111)",
                }
            ]
            payload = interface_energy(
                root, equilibration_frames=2, blocks=4, interface_metadata=metadata
            )
            self.assertEqual(payload["interfaces"], [])
            self.assertEqual(len(payload["skipped"]), 1)
            skipped = payload["skipped"][0]
            self.assertEqual(skipped["leaf"], "interface/300K/Ideal/N_Term/SiN_TiN_N-term")
            self.assertIn("polar", skipped["reason"].lower())
            self.assertIn("adhesion", skipped["reason"])

    def test_metadata_supplies_stacking_axis_n_interfaces_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            metadata = [
                {
                    "match": "interface/*/*/N_Term/*",
                    "stacking_axis": "a",  # |b x c| = 10 * 20 = 200 A^2
                    "n_interfaces": 4,
                    "orientation": "Si3N4(0001)/TiN(111)",
                    "termination": "N",
                }
            ]
            payload = interface_energy(
                root, equilibration_frames=2, blocks=4, interface_metadata=metadata
            )
            row = payload["interfaces"][0]
            self.assertEqual(row["stacking_axis"], "a")
            self.assertEqual(row["n_interfaces"], 4)
            self.assertEqual(row["orientation"], "Si3N4(0001)/TiN(111)")
            self.assertEqual(row["termination"], "N")
            self.assertAlmostEqual(row["interface_area_ang2"], 200.0)
            # excess = -100 - 4*(-12) - 4*(-10) = -12 eV ; /(4 * 200) ; *conv
            self.assertAlmostEqual(row["gamma_int_j_per_m2"], -12.0 / (4 * 200.0) * 16.02176634)
            self.assertIsNone(payload["n_interfaces"])
            self.assertEqual(payload["n_interfaces_source"], "campaign-metadata")

    def test_explicit_n_interfaces_argument_still_wins_over_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _dataset(Path(temporary))
            metadata = [{"match": "interface/*", "n_interfaces": 4}]
            payload = interface_energy(
                root,
                equilibration_frames=2,
                blocks=4,
                n_interfaces=2,
                interface_metadata=metadata,
            )
            self.assertEqual(payload["interfaces"][0]["n_interfaces"], 2)

    def test_empty_dataset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SafetyError):
                interface_energy(Path(temporary))


if __name__ == "__main__":
    unittest.main()
