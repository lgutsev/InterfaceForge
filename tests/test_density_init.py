"""Core safety, magnetism and provenance tests for ``iface vasp initialize-density``.

The neural_paw_dft worker is replaced by :class:`FakeRunner`; no ML stack is needed.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from density_init_fixtures import (
    NIO_INCAR,
    NIO_MAGMOM,
    STO_INCAR,
    STO_POSCAR,
    FakeRunner,
    write_run,
)

from interfaceforge.density_init import (
    INCAR_BACKUP_NAME,
    REPORT_NAME,
    STAGED_NAME,
    InferenceError,
    NeuralPawInitializer,
    initialize_density,
)
from interfaceforge.density_init.inputs import expand_vasp_list, grid_from_file, potcar_entries
from interfaceforge.errors import DependencyError, SafetyError
from interfaceforge.profile_cli import main
from interfaceforge.vasp import parse_incar

GRID = (24, 24, 24)


def _backend(runner: FakeRunner) -> NeuralPawInitializer:
    return NeuralPawInitializer(runner=runner, python="/fake/ndi/python")


def _snapshot(run: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(run.iterdir()) if path.is_file()}


class InputValidationTests(unittest.TestCase):
    def test_missing_required_inputs_are_refused_without_writing(self) -> None:
        for missing in ("INCAR", "POSCAR", "POTCAR", "KPOINTS"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                run = write_run(Path(tmp) / "run")
                (run / missing).unlink()
                before = _snapshot(run)
                runner = FakeRunner()
                with self.assertRaisesRegex(SafetyError, missing):
                    initialize_density(run, backend=_backend(runner), grid=GRID)
                self.assertEqual(_snapshot(run), before)
                self.assertEqual(runner.calls, [])

    def test_kpoints_optional_with_kspacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR + "KSPACING = 0.25\n", kpoints=False)
            result = initialize_density(run, backend=_backend(FakeRunner()), grid=GRID, dry_run=True)
            self.assertEqual(result["status"], "PLANNED")

    def test_started_run_and_wavecar_restart_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / "OUTCAR").write_text("running\n")
            with self.assertRaisesRegex(SafetyError, "already has VASP output"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            (run / "OUTCAR").unlink()
            (run / "INCAR").write_text(NIO_INCAR.replace("ISTART = 0\n", ""))
            (run / "WAVECAR").write_bytes(b"x" * 64)
            with self.assertRaisesRegex(SafetyError, "WAVECAR"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)

    def test_non_scf_icharg_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR + "ICHARG = 11\n")
            with self.assertRaisesRegex(SafetyError, "non-self-consistent"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)

    def test_grid_must_be_resolvable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            with self.assertRaisesRegex(SafetyError, "FFT grid"):
                initialize_density(run, backend=_backend(FakeRunner()))
            outcar = Path(tmp) / "OUTCAR.ref"
            outcar.write_text(" dimension x,y,z NGXF=    48 NGYF=    48 NGZF=    60\n")
            result = initialize_density(run, backend=_backend(FakeRunner()), grid_from=outcar, dry_run=True)
            self.assertEqual(result["grid"]["dims"], [48, 48, 60])
            (run / "INCAR").write_text(NIO_INCAR + "NGXF = 30\nNGYF = 30\nNGZF = 32\n")
            result = initialize_density(run, backend=_backend(FakeRunner()), dry_run=True)
            self.assertEqual(result["grid"], {"dims": [30, 30, 32], "source": "INCAR NGXF/NGYF/NGZF"})

    def test_input_parsers(self) -> None:
        self.assertEqual(expand_vasp_list("2*1.7 2*-1.7 4*0"), [1.7, 1.7, -1.7, -1.7, 0, 0, 0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            entries = potcar_entries(run / "POTCAR")
            self.assertEqual([e["symbol"] for e in entries], ["Ni_pv", "O"])
            self.assertEqual(entries[0]["projector_l"], [1, 1, 2, 2, 0, 0])
            self.assertEqual(entries[1]["zval"], 6.0)
            chgcar = run / "CHGCAR.test"
            chgcar.write_text((run / "POSCAR").read_text() + "\n   36   36   40\n 1 2 3\n")
            self.assertEqual(grid_from_file(chgcar), (36, 36, 40))


class DryRunAndBackendTests(unittest.TestCase):
    def test_dry_run_writes_nothing_and_reports_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            runner = FakeRunner()
            result = initialize_density(run, backend=_backend(runner), grid=GRID, dry_run=True)
            self.assertEqual(_snapshot(run), before)
            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual(result["status"], "PLANNED")
            self.assertIn(STAGED_NAME, result["would_write"])
            self.assertIn("CHGCAR", result["would_write"])
            self.assertEqual(runner.inference_calls, 0)
            self.assertTrue(result["backend_probe"]["available"])
            self.assertEqual(result["nelect"], {"value": 44.0, "source": "POTCAR ZVAL"})

    def test_backend_unavailable_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            runner = FakeRunner(available=False)
            with self.assertRaisesRegex(DependencyError, "neural_paw_dft"):
                initialize_density(run, backend=_backend(runner), grid=GRID)
            after = _snapshot(run)
            report = json.loads(after.pop(REPORT_NAME))
            after.pop("density_init.log", None)
            self.assertEqual(after, before)  # inputs untouched, no CHGCAR
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("DependencyError", report["error"])
            self.assertEqual(runner.inference_calls, 0)

    def test_inference_failure_rolls_back_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            with self.assertRaisesRegex(InferenceError, "CUDA out of memory"):
                initialize_density(run, backend=_backend(FakeRunner(fail="CUDA out of memory")), grid=GRID)
            after = _snapshot(run)
            report = json.loads(after.pop(REPORT_NAME))
            after.pop("density_init.log", None)
            self.assertEqual(after, before)
            self.assertEqual(report["status"], "FAILED")
            self.assertFalse(report["active"])
            self.assertFalse(any(p.name.startswith(".density_init") for p in run.iterdir()))

    def test_wrong_grid_from_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            with self.assertRaisesRegex(InferenceError, "grid"):
                initialize_density(run, backend=_backend(FakeRunner(wrong_grid=True)), grid=GRID)
            self.assertFalse((run / "CHGCAR").exists())
            self.assertFalse((run / STAGED_NAME).exists())

    def test_tampered_inputs_are_restored_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            original = (run / "INCAR").read_text()
            with self.assertRaisesRegex(SafetyError, "inputs changed"):
                initialize_density(run, backend=_backend(FakeRunner(tamper=run / "INCAR")), grid=GRID)
            self.assertEqual((run / "INCAR").read_text(), original)
            self.assertFalse((run / "CHGCAR").exists())

    def test_standard_backend_writes_only_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            result = initialize_density(run, backend="standard")
            after = _snapshot(run)
            after.pop(REPORT_NAME)
            self.assertEqual(after, before)
            self.assertEqual(result["status"], "STANDARD_START")


class SuccessAndFileSafetyTests(unittest.TestCase):
    def test_successful_generation_promotes_and_sets_only_icharg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            original_incar = (run / "INCAR").read_text()
            result = initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            self.assertEqual(result["status"], "PROMOTED")
            self.assertTrue(result["active"])
            self.assertTrue((run / "CHGCAR").is_file())
            self.assertEqual((run / "CHGCAR").read_bytes(), (run / STAGED_NAME).read_bytes())
            self.assertEqual((run / INCAR_BACKUP_NAME).read_text(), original_incar)
            new_incar = (run / "INCAR").read_text()
            self.assertTrue(new_incar.startswith(original_incar))  # every original line untouched
            self.assertEqual(parse_incar(run / "INCAR"), {**parse_incar(run / INCAR_BACKUP_NAME), "ICHARG": "1"})
            self.assertEqual(result["incar_changes"], [
                {"tag": "ICHARG", "before": None, "after": "1", "reason": "read promoted CHGCAR"}
            ])
            self.assertTrue(result["inputs"]["unchanged_except_incar_icharg"])
            self.assertFalse((run / ".density_init.lock").exists())

    def test_existing_icharg_is_replaced_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR.replace("ISTART = 0\n", "ISTART = 0\nICHARG = 2\n"))
            initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            lines = (run / "INCAR").read_text().splitlines()
            self.assertIn("ICHARG = 1", lines)
            self.assertNotIn("ICHARG = 2", lines)
            self.assertEqual(len(lines), len((run / INCAR_BACKUP_NAME).read_text().splitlines()))

    def test_existing_foreign_chgcar_is_never_replaced_silently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / "CHGCAR").write_text("converged density from a previous job\n")
            incar = (run / "INCAR").read_text()
            result = initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            self.assertEqual(result["status"], "STAGED")
            self.assertFalse(result["active"])
            self.assertEqual((run / "CHGCAR").read_text(), "converged density from a previous job\n")
            self.assertEqual((run / "INCAR").read_text(), incar)
            self.assertTrue((run / STAGED_NAME).is_file())
            self.assertTrue(any("existing CHGCAR kept" in w for w in result["warnings"]))

    def test_staged_run_is_promoted_once_the_foreign_chgcar_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / "CHGCAR").write_text("foreign\n")
            runner = FakeRunner()
            self.assertEqual(initialize_density(run, backend=_backend(runner), grid=GRID)["status"], "STAGED")
            again = initialize_density(run, backend=_backend(runner), grid=GRID)
            self.assertEqual(again["mode"], "already-initialized")  # still blocked: nothing to do
            (run / "CHGCAR").unlink()
            promoted = initialize_density(run, backend=_backend(runner), grid=GRID)
            self.assertEqual(promoted["status"], "PROMOTED")
            self.assertEqual(parse_incar(run / "INCAR")["ICHARG"], "1")

    def test_overwrite_backs_up_foreign_chgcar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / "CHGCAR").write_text("user density\n")
            result = initialize_density(run, backend=_backend(FakeRunner()), grid=GRID, overwrite=True)
            self.assertEqual(result["status"], "PROMOTED")
            backup = run / result["files"]["backed_up_chgcar"]
            self.assertEqual(backup.read_text(), "user density\n")
            self.assertNotEqual((run / "CHGCAR").read_text(), "user density\n")

    def test_stage_only_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            incar = (run / "INCAR").read_text()
            result = initialize_density(run, backend=_backend(FakeRunner()), grid=GRID, stage_only=True)
            self.assertEqual(result["status"], "STAGED")
            self.assertFalse((run / "CHGCAR").exists())
            self.assertEqual((run / "INCAR").read_text(), incar)

    def test_rerun_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            runner = FakeRunner()
            initialize_density(run, backend=_backend(runner), grid=GRID)
            snapshot = _snapshot(run)
            again = initialize_density(run, backend=_backend(runner), grid=GRID)
            self.assertEqual(again["mode"], "already-initialized")
            self.assertEqual(runner.inference_calls, 1)
            self.assertEqual(_snapshot(run), snapshot)
            forced = initialize_density(run, backend=_backend(runner), grid=GRID, force=True)
            self.assertEqual(forced["status"], "PROMOTED")  # our own CHGCAR may be replaced
            self.assertEqual(runner.inference_calls, 2)
            self.assertNotIn("backed_up_chgcar", forced["files"])
            self.assertEqual(parse_incar(run / "INCAR")["ICHARG"], "1")

    def test_changed_structure_demotes_stale_density_before_regenerating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            runner = FakeRunner()
            initialize_density(run, backend=_backend(runner), grid=GRID)
            (run / "POSCAR").write_text((run / "POSCAR").read_text().replace("0.25 0.25 0.25", "0.26 0.25 0.25"))
            result = initialize_density(run, backend=_backend(runner), grid=GRID)
            self.assertEqual(runner.inference_calls, 2)
            self.assertEqual(result["status"], "PROMOTED")
            stale = run / result["files"]["demoted_stale"]
            self.assertTrue(stale.is_file())
            self.assertIn("0.26 0.25 0.25", (run / "CHGCAR").read_text())

    def test_concurrent_lock_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / ".density_init.lock").write_text("123\n")
            with self.assertRaisesRegex(SafetyError, "lock"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            self.assertFalse((run / "CHGCAR").exists())


class MagnetismTests(unittest.TestCase):
    def test_signed_afm_magmom_is_preserved_exactly_with_charge_only_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            runner = FakeRunner()
            result = initialize_density(run, backend=_backend(runner), grid=GRID, magmom_source="incar")
            request = runner.requests[0]
            self.assertFalse(request["spin_channel"])  # CHGCAR carries no magnetization
            self.assertFalse(request["use_initializer_moments"])  # CHGNet never consulted
            self.assertIsNone(request["site_moments"])
            self.assertEqual(parse_incar(run / "INCAR")["MAGMOM"], NIO_MAGMOM)
            self.assertIn(f"MAGMOM = {NIO_MAGMOM}\n", (run / "INCAR").read_text())
            magnetism = result["magnetism"]
            self.assertEqual(magnetism["order"], "mixed-sign")
            self.assertEqual(magnetism["incar_magmom"], [1.7, -1.7, 0.0, 0.0])
            self.assertTrue(magnetism["automatic_magnetic_initialization"].startswith("overridden"))
            self.assertIn("INCAR MAGMOM", magnetism["initial_moments_in_vasp_from"])
            self.assertFalse(magnetism["spin_channel_written"])
            self.assertIsNone(magnetism["initializer_moments"])

    def test_model_spin_channel_refused_for_afm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            with self.assertRaisesRegex(SafetyError, "net moment"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID, spin_channel="model")

    def test_initializer_moments_refused_for_mixed_sign_incar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            with self.assertRaisesRegex(SafetyError, "mixed signs"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID, magmom_source="initializer")
            self.assertEqual(_snapshot(run), before)

    def test_sign_uniform_incar_moments_constrain_the_model_spin_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR.replace(NIO_MAGMOM, "2*1.7 2*0"))
            runner = FakeRunner()
            result = initialize_density(run, backend=_backend(runner), grid=GRID)
            request = runner.requests[0]
            self.assertTrue(request["spin_channel"])
            self.assertEqual(request["site_moments"], [1.7, 1.7, 0.0, 0.0])
            self.assertFalse(request["use_initializer_moments"])
            self.assertTrue(result["magnetism"]["spin_channel_written"])

    def test_spin_channel_off_forces_charge_only_even_for_ferromagnets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR.replace(NIO_MAGMOM, "2*1.7 2*0"))
            runner = FakeRunner()
            initialize_density(run, backend=_backend(runner), grid=GRID, spin_channel="off")
            self.assertFalse(runner.requests[0]["spin_channel"])

    def test_explicit_initializer_moments_are_reported_as_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR.replace(f"MAGMOM = {NIO_MAGMOM}\n", ""))
            runner = FakeRunner(moments=[1.1, 1.1, 0.0, 0.0])
            result = initialize_density(run, backend=_backend(runner), grid=GRID, magmom_source="initializer")
            self.assertTrue(runner.requests[0]["use_initializer_moments"])
            self.assertEqual(result["magnetism"]["initializer_moments"], [1.1, 1.1, 0.0, 0.0])
            self.assertTrue(result["magnetism"]["automatic_magnetic_initialization"].startswith("used"))
            self.assertNotIn("MAGMOM", parse_incar(run / "INCAR"))  # INCAR never rewritten

    def test_magmom_length_and_noncollinear_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", incar=NIO_INCAR.replace(NIO_MAGMOM, "1.7 -1.7"))
            with self.assertRaisesRegex(SafetyError, "MAGMOM has 2 entries"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            (run / "INCAR").write_text(NIO_INCAR + "LSORBIT = .TRUE.\n")
            with self.assertRaisesRegex(SafetyError, "Non-collinear"):
                initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)

    def test_nonmagnetic_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(
                Path(tmp) / "run", poscar=STO_POSCAR, incar=STO_INCAR, potcar_symbols=("Sr_sv", "Ti_pv", "O")
            )
            runner = FakeRunner()
            result = initialize_density(run, backend=_backend(runner), grid=GRID)
            self.assertEqual(result["magnetism"]["order"], "non-spin-polarized")
            self.assertFalse(runner.requests[0]["spin_channel"])
            with self.assertRaisesRegex(SafetyError, "ISPIN = 2"):
                initialize_density(run, backend=_backend(runner), grid=GRID, spin_channel="model", force=True)


class ProvenanceTests(unittest.TestCase):
    def test_json_provenance_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            initialize_density(run, backend=_backend(FakeRunner()), grid=GRID)
            report = json.loads((run / REPORT_NAME).read_text())
            self.assertEqual(report["format"], "interfaceforge-density-init")
            self.assertEqual(report["status"], "PROMOTED")
            self.assertEqual(report["backend"], "neural-paw")
            info = report["backend_info"]
            self.assertEqual(info["package"], "neural_paw_dft")
            self.assertEqual(info["package_version"], "0.1.0")
            self.assertEqual(info["package_commit"], "abc1234")
            self.assertEqual(info["models"]["electrafi"]["name"], "electrafi_total")
            self.assertIn("version", report["interfaceforge"])
            self.assertIn("commit", report["interfaceforge"])
            self.assertEqual(len(report["structure"]["geometry_sha256"]), 64)
            self.assertEqual(report["structure"]["composition"], {"Ni": 2, "O": 2})
            self.assertEqual(report["magnetism"]["magmom_source"], "incar")
            self.assertEqual(report["files"]["generated"], [STAGED_NAME, "CHGCAR"])
            self.assertEqual(report["grid"], {"dims": list(GRID), "source": "explicit --grid"})
            self.assertEqual(report["timing"]["inference_s"], 1.25)
            self.assertGreaterEqual(report["timing"]["total_s"], 0.0)
            self.assertIn("warnings", report)
            for name in ("INCAR", "POSCAR", "POTCAR", "KPOINTS"):
                self.assertEqual(len(report["inputs"]["sha256_before"][name]), 64)


class CliTests(unittest.TestCase):
    """Drive the real CLI and the real worker script (neural_paw_dft is absent here)."""

    def _cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_cli_dry_run_reports_unavailable_backend_via_real_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            code, out, _ = self._cli(
                ["vasp", "initialize-density", str(run), "--grid", "24", "24", "24", "--dry-run",
                 "--ndi-python", sys.executable]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(payload["status"], "PLANNED")
            self.assertEqual(_snapshot(run), before)
            try:
                import neural_paw_dft  # noqa: F401
            except ImportError:
                self.assertFalse(payload["backend_probe"]["available"])

    def test_cli_real_run_without_backend_exits_nonzero_and_keeps_inputs(self) -> None:
        try:
            import neural_paw_dft  # noqa: F401

            self.skipTest("neural_paw_dft installed; covered by the integration test")
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            before = _snapshot(run)
            code, _, err = self._cli(
                ["vasp", "initialize-density", str(run), "--grid", "24", "24", "24", "--ndi-python", sys.executable]
            )
            self.assertEqual(code, 2)
            self.assertIn("unavailable", err)
            after = _snapshot(run)
            self.assertEqual(json.loads(after.pop(REPORT_NAME))["status"], "FAILED")
            after.pop("density_init.log", None)
            self.assertEqual(after, before)

    def test_probe_command_exit_code(self) -> None:
        env_python = os.environ.get("IFACE_NDI_PYTHON")
        code, out, _ = self._cli(["vasp", "density-init-probe", "--ndi-python", env_python or sys.executable])
        payload = json.loads(out)
        self.assertEqual(code, 0 if payload["probe"]["available"] else 3)


if __name__ == "__main__":
    unittest.main()
