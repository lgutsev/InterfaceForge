from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.errors import SafetyError
from interfaceforge.slab_alignment import parse_incar
from interfaceforge.slab_tight_scf import (
    prepare_tight_scf,
    rewrite_incar,
    scf_diagnostics_from_outcar,
    selective_free_mask,
)

INCAR = """GGA = PE

#Restart Flags Above
LVHAR = .TRUE.
ISMEAR=0
IBRION = 2
LASPH = TRUE
ENCUT = 520
LASPH = TRUE
IVDW = 11
LREAL = AUTO
LWAVE = TRUE
ISIF = 2; NSW = 99   # relaxation
NCORE =  4
LDIPOL = .TRUE.
DIPOL  = 0.5 0.5 0.5
IDIPOL = 1
"""

POSCAR = """slab
1.0
40 0 0
0 10 0
0 0 10
Pb I
1 1
Direct
0.40 0.5 0.5
0.60 0.5 0.5
"""

POTCAR = "  VRHFIN =Pb: 6s5d6p\n  VRHFIN =I: 5s5p\n"


def _outcar(
    *,
    converged_steps: int,
    unconverged_last: bool = False,
    finished: bool = True,
    nsw: int = 99,
    reached: bool = True,
    forces: list[tuple[float, float, float]] | None = None,
) -> str:
    lines = [
        f"   NSW    =     {nsw}    number of steps for IOM",
        "   IBRION =      2    ionic relax: 0-MD 1-quasi-New 2-CG",
        "   ISTART =      1    job   : 0-new  1-cont  2-samecut",
        "   ICHARG =      0    charge: 1-file 2-atom 10-const",
        "   NELM   =     60;   NELMIN=  2; NELMDL=  0     # of ELM steps",
        "   EDIFF  = 0.1E-03   stopping-criterion for ELM",
        "   EDIFFG = 0.1E-02   stopping-criterion for IOM",
        "     AMIN     =   0.10",
        "|     large. This can spoil convergence since charge sloshing might occur     |",
    ]
    for step in range(1, converged_steps + 1):
        lines.append(f"----- Iteration {step:6d}(   1)  -----")
        lines.append(f"----- Iteration {step:6d}(   2)  -----")
        lines.append("------------------------ aborting loop because EDIFF is reached -----")
    if unconverged_last:
        step = converged_steps + 1
        lines.append(f"----- Iteration {step:6d}(  59)  -----")
        lines.append(f"----- Iteration {step:6d}(  60)  -----")
    lines.append("|     The minimum charge density times volume of the cell along the axis      |")
    lines.append(" POSITION                                       TOTAL-FORCE (eV/Angst)")
    lines.append(" " + "-" * 83)
    for fx, fy, fz in forces or [(0.01, 0.0, 0.0), (0.0, 0.02, 0.0)]:
        lines.append(f"     16.40000      5.00000      5.00000      {fx:10.6f}   {fy:10.6f}   {fz:10.6f}")
    lines.append(" " + "-" * 83)
    lines.append("    total drift:                                0.004443     -0.155348      0.025823")
    if reached:
        lines.append(" reached required accuracy - stopping structural energy minimisation")
    if finished:
        lines.append(" General timing and accounting informations for this job:")
    return "\n".join(lines) + "\n"


def _calc(root: Path, name: str, **outcar_kwargs: object) -> Path:
    folder = root / name
    folder.mkdir()
    (folder / "INCAR").write_text(INCAR, encoding="utf-8")
    (folder / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (folder / "CONTCAR").write_text(POSCAR.replace("0.40", "0.41"), encoding="utf-8")
    (folder / "KPOINTS").write_text("Gamma\n0\nG\n1 3 3\n", encoding="utf-8")
    (folder / "POTCAR").write_text(POTCAR, encoding="utf-8")
    (folder / "WAVECAR").write_bytes(b"\x00" * 64)
    (folder / "OUTCAR").write_text(_outcar(**{"converged_steps": 3, **outcar_kwargs}), encoding="utf-8")
    return folder


def _row(name: str, flatness: str, **extra: object) -> dict[str, object]:
    return {
        "folder": name,
        "reference": "Ref",
        "flatness_status": flatness,
        "vasp_vacuum_warning": "",
        "axis": "x",
        "selected_swing_eV": 0.09,
        "suggested_DIPOL_normal": 0.48,
        **extra,
    }


def _family(root: Path) -> None:
    for name in ("Ref", "Ref_A", "Ref_Ok", "Ref_Tilted"):
        _calc(root, name)
    rows = [
        _row("Ref", "OK"),
        _row("Ref_A", "SUSPECT_FLATNESS", vasp_vacuum_warning="VACUUM_CHARGE_DENSITY_TOO_LARGE"),
        _row("Ref_Ok", "OK"),
        _row("Ref_Tilted", "FAILED_ANALYSIS", error="tilted"),
    ]
    (root / "band_edge_alignment.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (root / "slab_alignment_fapi.json").write_text('{"axis": "x", "side": "low-x"}', encoding="utf-8")


class RewriteIncarTests(unittest.TestCase):
    def test_overrides_semicolons_duplicates_and_appends(self) -> None:
        text, changes = rewrite_incar(INCAR, {"NSW": "0", "LASPH": ".TRUE.", "AMIN": "0.01"})
        self.assertIn("ISIF = 2; NSW = 0  # relaxation", text)
        self.assertEqual(text.count("LASPH"), 1)
        self.assertTrue(text.rstrip().endswith("AMIN = 0.01"))
        tags = {item["tag"]: item for item in changes}
        self.assertEqual(tags["NSW"]["old"], "99")
        self.assertEqual(tags["AMIN"]["old"], "(unset)")
        self.assertNotIn("LASPH", tags)  # TRUE == .TRUE.
        self.assertIn("ENCUT = 520", text)
        self.assertIn("LREAL = AUTO", text)


class OutcarDiagnosticsTests(unittest.TestCase):
    def test_parses_settings_and_convergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_outcar(converged_steps=4), encoding="utf-8")
            diag = scf_diagnostics_from_outcar(path)
            self.assertTrue(diag.finished)
            self.assertEqual(diag.ionic_steps, 4)
            self.assertEqual(diag.unconverged_scf_steps, [])
            self.assertTrue(diag.final_scf_converged)
            self.assertAlmostEqual(diag.EDIFF, 1e-4)
            self.assertEqual(diag.NELM, 60)
            self.assertAlmostEqual(diag.AMIN, 0.10)
            self.assertEqual((diag.ISTART, diag.ICHARG), (1, 0))
            self.assertTrue(diag.charge_sloshing_warning)
            self.assertEqual(diag.vacuum_warning, "VACUUM_CHARGE_DENSITY_TOO_LARGE")

    def test_flags_step_that_exhausted_nelm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_outcar(converged_steps=2, unconverged_last=True), encoding="utf-8")
            diag = scf_diagnostics_from_outcar(path)
            self.assertEqual(diag.unconverged_scf_steps, [3])
            self.assertFalse(diag.final_scf_converged)


class IonicConvergenceTests(unittest.TestCase):
    def test_nsw_limit_and_high_forces_on_free_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(
                _outcar(converged_steps=3, nsw=3, reached=False, forces=[(0.9, 0.0, 0.0), (0.0, 0.08, 0.0)]),
                encoding="utf-8",
            )
            diag = scf_diagnostics_from_outcar(path)
            self.assertTrue(diag.hit_nsw_limit)
            self.assertFalse(diag.ionic_converged)
            self.assertAlmostEqual(diag.final_max_force_eV_per_A, 0.9)
            self.assertAlmostEqual(diag.EDIFFG, 1e-3)
            # The first atom is fixed by selective dynamics, so its force is ignored.
            diag = scf_diagnostics_from_outcar(path, free_mask=[False, True])
            self.assertAlmostEqual(diag.final_max_force_eV_per_A, 0.08)
            self.assertEqual(diag.force_atoms, "free")

    def test_converged_relaxation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_outcar(converged_steps=3), encoding="utf-8")
            diag = scf_diagnostics_from_outcar(path)
            self.assertTrue(diag.ionic_converged)
            self.assertFalse(diag.hit_nsw_limit)
            self.assertAlmostEqual(diag.final_max_force_eV_per_A, 0.02)

    def test_selective_free_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CONTCAR"
            path.write_text(POSCAR.replace("Pb I", "Sn I"), encoding="utf-8")
            self.assertIsNone(selective_free_mask(path))
            path.write_text(
                POSCAR.replace(
                    "Direct\n0.40 0.5 0.5\n0.60 0.5 0.5",
                    "Selective dynamics\nDirect\n0.40 0.5 0.5 F F F\n0.60 0.5 0.5 T T F",
                ),
                encoding="utf-8",
            )
            self.assertEqual(selective_free_mask(path), [False, True])


class PrepareTightScfTests(unittest.TestCase):
    def test_flagged_folder_and_reference_control_are_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root)
            parent_incar = (root / "Ref_A" / "INCAR").read_text(encoding="utf-8")
            result = prepare_tight_scf(root, config="slab_alignment_fapi.json")
            actions = {entry["folder"]: (entry["role"], entry["action"]) for entry in result["plan"]}
            self.assertEqual(actions["Ref_A"], ("flagged", "PREPARED"))
            self.assertEqual(actions["Ref"], ("reference_control", "PREPARED"))
            self.assertEqual(actions["Ref_Ok"][1], "SKIPPED")
            self.assertEqual(actions["Ref_Tilted"][1], "SKIPPED")

            child = root / "tight_scf" / "Ref_A"
            incar = (child / "INCAR").read_text(encoding="utf-8")
            for expected in (
                "NSW = 0",
                "IBRION = -1",
                "EDIFF = 1E-07",
                "NELM = 200",
                "AMIN = 0.01",
                "ISTART = 1",
                "ICHARG = 0",
                "LVACPOTAV = .TRUE.",
                "LCHARG = .TRUE.",
                "IDIPOL = 1",
                "LREAL = AUTO",
                "IVDW = 11",
            ):
                self.assertIn(expected, incar)
            self.assertNotIn("ICHARG = 11", incar)
            self.assertEqual(parse_incar(child / "INCAR")["DIPOL"], [0.5, 0.5, 0.5])
            self.assertIn("0.41", (child / "POSCAR").read_text(encoding="utf-8"))
            for name in ("KPOINTS", "POTCAR", "WAVECAR", "INCAR.parent", "TIGHT_SCF_PROVENANCE.json"):
                self.assertTrue((child / name).is_file(), name)
            provenance = json.loads((child / "TIGHT_SCF_PROVENANCE.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(provenance["parent_scf"]["EDIFF"], 1e-4)
            self.assertTrue((root / "tight_scf" / "slab_alignment_fapi.json").is_file())
            self.assertTrue((root / "tight_scf" / "tight_scf_plan.tsv").is_file())
            # Parents are untouched.
            self.assertEqual((root / "Ref_A" / "INCAR").read_text(encoding="utf-8"), parent_incar)

            again = prepare_tight_scf(root, config="slab_alignment_fapi.json")
            self.assertEqual(again["counts"].get("SKIPPED_EXISTS"), 2)

    def test_select_all_without_wavecar_and_suggested_dipol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root)
            result = prepare_tight_scf(
                root, select="all", wavecar="none", dipol="suggested", with_references=False
            )
            self.assertEqual(result["counts"], {"PREPARED": 4})
            incar = (root / "tight_scf" / "Ref_Tilted" / "INCAR").read_text(encoding="utf-8")
            self.assertIn("ISTART = 0", incar)
            self.assertIn("ICHARG = 2", incar)
            self.assertFalse((root / "tight_scf" / "Ref_Tilted" / "WAVECAR").exists())
            self.assertEqual(parse_incar(root / "tight_scf" / "Ref_Tilted" / "INCAR")["DIPOL"], [0.48, 0.5, 0.5])

    def test_unfinished_parent_or_species_mismatch_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root)
            (root / "Ref_A" / "OUTCAR").write_text(_outcar(converged_steps=2, finished=False), encoding="utf-8")
            (root / "Ref" / "POTCAR").write_text("  VRHFIN =I: x\n  VRHFIN =Pb: y\n", encoding="utf-8")
            result = prepare_tight_scf(root)
            blocked = {entry["folder"]: entry["reason"] for entry in result["plan"] if entry["action"] == "BLOCKED"}
            self.assertIn("incomplete", blocked["Ref_A"])
            self.assertIn("POTCAR order", blocked["Ref"])
            self.assertFalse((root / "tight_scf" / "Ref_A").exists())

    def test_unrelaxed_parent_warns_or_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root)
            (root / "Ref_A" / "OUTCAR").write_text(
                _outcar(converged_steps=3, nsw=3, reached=False, forces=[(0.2, 0.0, 0.0)]), encoding="utf-8"
            )
            result = prepare_tight_scf(root, dry_run=True)
            entry = next(item for item in result["plan"] if item["folder"] == "Ref_A")
            self.assertEqual(entry["action"], "WOULD_PREPARE")
            self.assertTrue(entry["parent_hit_nsw_limit"])
            self.assertTrue(any(w.startswith("PARENT_HIT_NSW_LIMIT") for w in entry["warnings"]))
            self.assertTrue(any("energy criterion" in w for w in entry["warnings"]))
            self.assertEqual(result["geometry_warnings"], 1)
            self.assertIn("warn: PARENT_HIT_NSW_LIMIT", (root / "tight_scf_plan.txt").read_text(encoding="utf-8"))

            blocked = prepare_tight_scf(root, require_relaxed=True)
            entry = next(item for item in blocked["plan"] if item["folder"] == "Ref_A")
            self.assertEqual(entry["action"], "BLOCKED")
            self.assertIn("require-relaxed", entry["reason"])

    def test_dry_run_writes_only_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _family(root)
            result = prepare_tight_scf(root, dry_run=True)
            self.assertEqual(result["counts"]["WOULD_PREPARE"], 2)
            self.assertFalse((root / "tight_scf").exists())
            self.assertTrue((root / "tight_scf_plan.txt").is_file())

    def test_missing_audit_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SafetyError):
                prepare_tight_scf(tmp)


if __name__ == "__main__":
    unittest.main()
