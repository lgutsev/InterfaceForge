from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.cli import main
from interfaceforge.geometry import batch_slab_vacuum, extend_slab_vacuum, slab_vacuum

try:
    from ase import Atoms
    from ase.io import write as ase_write

    HAVE_ASE = True
except ImportError:  # pragma: no cover
    HAVE_ASE = False


def _slab(vacuum: float, *, thickness: float = 6.0, offset: float = 0.0) -> Atoms:
    """A filled toy slab: 2x2 in-plane, dense in c, sitting `offset` Å above z=0."""

    c = thickness + vacuum
    zs = np.linspace(offset, offset + thickness, 5) % c
    positions = [(x, y, z) for x in (0.0, 3.0) for y in (0.0, 3.0) for z in zs]
    return Atoms(f"Ni{len(positions)}", positions=positions, cell=[6.0, 6.0, c], pbc=True)


@unittest.skipUnless(HAVE_ASE, "ASE not installed")
class SlabVacuumTests(unittest.TestCase):
    def _write(self, atoms: Atoms, path: Path) -> Path:
        ase_write(str(path), atoms, format="vasp", direct=True, vasp5=True)
        return path

    def test_gap_to_image_is_position_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a").mkdir()
            (Path(tmp) / "b").mkdir()
            centred = slab_vacuum(self._write(_slab(13.0, offset=6.5), Path(tmp) / "a" / "POSCAR"))
            skewed = slab_vacuum(self._write(_slab(13.0, offset=0.5), Path(tmp) / "b" / "POSCAR"))
            self.assertAlmostEqual(centred["vacuum_a"], 13.0, places=3)
            self.assertAlmostEqual(skewed["vacuum_a"], 13.0, places=3)  # same real headroom
            self.assertEqual(centred["axis"], "c")

    def test_auto_axis_picks_the_normal_on_a_real_slab(self) -> None:
        from ase.build import fcc111

        with tempfile.TemporaryDirectory() as tmp:
            poscar = self._write(fcc111("Ni", size=(3, 3, 4), vacuum=9.0), Path(tmp) / "POSCAR")
            report = slab_vacuum(poscar)
            self.assertEqual(report["axis"], "c")
            self.assertAlmostEqual(report["vacuum_a"], 18.0, delta=0.5)

    def test_wrapped_slab_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            atoms = _slab(6.0, thickness=6.0)  # c = 12
            atoms.positions[:, 2] = (atoms.positions[:, 2] + 8.0) % 12.0  # straddle the boundary
            report = slab_vacuum(self._write(atoms, Path(tmp) / "POSCAR"))
            self.assertTrue(report["wrapped"])
            self.assertAlmostEqual(report["vacuum_a"], 6.0, places=3)

    def test_extend_grows_only_the_normal_and_keeps_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(4.0, offset=2.0), Path(tmp) / "POSCAR")
            before = slab_vacuum(src)
            out = Path(tmp) / "POSCAR.ext"
            extend_slab_vacuum(src, out, vacuum=18.0)
            after = slab_vacuum(out)
            self.assertAlmostEqual(after["vacuum_a"], 18.0, places=2)
            self.assertAlmostEqual(after["slab_span_a"], before["slab_span_a"], places=2)

    def test_extend_leaves_a_roomier_cell_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(25.0, offset=12.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "POSCAR.ext"
            summary = extend_slab_vacuum(src, out, vacuum=18.0)
            self.assertAlmostEqual(summary["vacuum_after_a"], 25.0, places=1)

    def test_batch_audit_reports_total_headroom_not_box_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, (vac, off) in {
                "roomy": (20.0, 0.5),  # skewed against z=0 but 20 A real headroom -> PASS
                "tight": (7.0, 3.0),
            }.items():
                (root / "Step2_300K" / name).mkdir(parents=True)
                self._write(_slab(vac, offset=off), root / "Step2_300K" / name / "POSCAR")

            report = batch_slab_vacuum(root, min_vacuum=12.0)
            rows = {r["path"].replace("\\", "/"): r for r in report["rows"]}
            self.assertEqual(rows["Step2_300K/roomy/POSCAR"]["status"], "PASS")
            self.assertEqual(rows["Step2_300K/tight/POSCAR"]["status"], "THIN")
            self.assertEqual(report["thin"], 1)

            # dry run: plans the stretch, writes nothing
            dry = batch_slab_vacuum(root, min_vacuum=12.0, extend=18.0)
            self.assertEqual(dry["mode"], "dry-run")
            self.assertEqual(dry["extended"], 0)
            tight = next(r for r in dry["rows"] if "tight" in r["path"])
            self.assertEqual(tight["would_be_a"], 18.0)
            self.assertEqual(batch_slab_vacuum(root, min_vacuum=12.0)["thin"], 1)

            # execute
            done = batch_slab_vacuum(root, min_vacuum=12.0, extend=18.0, execute=True)
            self.assertEqual(done["mode"], "extended")
            self.assertEqual(done["extended"], 1)
            self.assertEqual(batch_slab_vacuum(root, min_vacuum=12.0)["thin"], 0)

    def test_cli_audit_flags_thin_and_suggests_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(6.0, offset=2.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "out.json"
            rc = main(["vasp", "geom", "vacuum", str(src), "--min-vacuum", "12", "--summary", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "THIN")
            self.assertIn("--extend", payload["fix"])

    def test_cli_extend_dry_run_then_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(6.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "dry.json"

            # bare --extend on one file = dry run, no write
            self.assertEqual(
                main(["vasp", "geom", "vacuum", str(src), "--extend", "18", "--summary", str(out)]), 0
            )
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["mode"], "dry-run")
            self.assertAlmostEqual(slab_vacuum(src)["vacuum_a"], 6.0, places=2)

            # --execute overwrites in place
            self.assertEqual(
                main(["vasp", "geom", "vacuum", str(src), "--extend", "18", "--execute"]), 0
            )
            self.assertGreaterEqual(slab_vacuum(src)["vacuum_a"], 17.9)

    def test_cli_extend_to_named_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write(_slab(6.0), Path(tmp) / "POSCAR")
            out = Path(tmp) / "POSCAR.big"
            self.assertEqual(
                main(["vasp", "geom", "vacuum", str(src), "--extend", "18", "-o", str(out)]), 0
            )
            self.assertGreaterEqual(slab_vacuum(out)["vacuum_a"], 17.9)


if __name__ == "__main__":
    unittest.main()
