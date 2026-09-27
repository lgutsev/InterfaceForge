"""POTCAR provenance, requested-vs-executed status, launcher fallback records and
same-POTCAR enforcement for the optional density initializer."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml
from density_init_fixtures import (
    NIO_MAGMOM,
    STO_INCAR,
    STO_POSCAR,
    FakeRunner,
    potcar,
    write_run,
)
from test_density_init_bench import finish_run

from interfaceforge.density_init import NeuralPawInitializer, initialize_density
from interfaceforge.density_init.benchmark import compare_benchmark, prepare_benchmark
from interfaceforge.density_init.launch import FALLBACK_NAME, wrap_launcher_with_density_init
from interfaceforge.density_init.potcar import potcar_provenance, read_potcar_definitions
from interfaceforge.density_init.status import density_init_run_status
from interfaceforge.errors import SafetyError
from interfaceforge.step1_status import step1_status
from interfaceforge.vasp import parse_incar, prepare_step1_series, wrap_launcher_with_precondition

GRID = (24, 24, 24)
# Excerpts of the production and Materials Project-compatible POTCAR_gen maps.
PRODUCTION_DEFS = "# production\nNi|Ni\nO|O\nSr|Sr_sv\nTi|Ti_sv\n"
MP_DEFS = "Ni|Ni_pv\r\nO|O\r\n\r\nSr|Sr_sv\r\nTi|Ti_pv\r\n"


def _defs(root: Path) -> tuple[Path, Path]:
    production, mp = root / "POTCAR_DEFS.txt", root / "POTCAR_DEFS_MP.txt"
    production.write_text(PRODUCTION_DEFS)
    mp.write_bytes(MP_DEFS.encode())
    return production, mp


def _backend(runner: FakeRunner | None = None) -> NeuralPawInitializer:
    return NeuralPawInitializer(runner=runner or FakeRunner(), python="/fake/ndi/python")


class PotcarProvenanceTests(unittest.TestCase):
    def test_definitions_are_parsed_like_potcar_gen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, mp = _defs(Path(tmp))
            self.assertEqual(read_potcar_definitions(production)["Ni"], "Ni")
            self.assertEqual(read_potcar_definitions(mp), {"Ni": "Ni_pv", "O": "O", "Sr": "Sr_sv", "Ti": "Ti_pv"})
            bad = Path(tmp) / "bad.txt"
            bad.write_text("Ni|\n")
            with self.assertRaisesRegex(SafetyError, "invalid POTCAR mapping"):
                read_potcar_definitions(bad)

    def test_actual_potcar_is_recorded_and_checked_against_the_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, mp = _defs(Path(tmp))
            run = write_run(Path(tmp) / "run")  # Ni_pv + O
            plain = potcar_provenance(run / "POTCAR", ["Ni", "Ni", "O", "O"])
            self.assertEqual(plain["potcar_variants"], {"Ni": "Ni_pv", "O": "O"})
            self.assertEqual(len(plain["potcar_sha256"]), 64)
            self.assertIsNone(plain["potcar_definitions"])
            declared = potcar_provenance(
                run / "POTCAR", ["Ni", "O"], definitions=mp, generator="/home/user/bin/POTCAR_gen"
            )
            self.assertEqual(declared["potcar_definitions"], str(mp.resolve()))
            self.assertEqual(declared["declared_variants"], {"Ni": "Ni_pv", "O": "O"})
            self.assertTrue(declared["consistent_with_definitions"])
            self.assertEqual(declared["potcar_generator"], "/home/user/bin/POTCAR_gen")
            with self.assertRaisesRegex(SafetyError, "Ni: declared 'Ni', POTCAR has 'Ni_pv'"):
                potcar_provenance(run / "POTCAR", ["Ni", "O"], definitions=production)

    def test_initialize_density_records_provenance_and_refuses_a_wrong_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, mp = _defs(Path(tmp))
            run = write_run(Path(tmp) / "run")
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            runner = FakeRunner()
            with self.assertRaisesRegex(SafetyError, "does not match the declared POTCAR definitions"):
                initialize_density(run, backend=_backend(runner), grid=GRID, potcar_definitions=production)
            self.assertEqual({path.name: path.read_bytes() for path in run.iterdir()}, before)
            self.assertEqual(runner.inference_calls, 0)
            result = initialize_density(
                run, backend=_backend(), grid=GRID, potcar_definitions=mp, potcar_generator="POTCAR_gen"
            )
            self.assertEqual(result["status"], "PROMOTED")
            report = json.loads((run / "density_init.json").read_text())
            self.assertEqual(report["potcar"]["potcar_variants"], {"Ni": "Ni_pv", "O": "O"})
            self.assertEqual(report["potcar"]["potcar_sha256"], report["inputs"]["sha256_before"]["POTCAR"])
            self.assertEqual(report["potcar"]["potcar_definitions"], str(mp.resolve()))
            # NiO AFM-II with the Ni_pv mapping: charge-only seed, signed MAGMOM byte-identical
            self.assertFalse(report["magnetism"]["spin_channel_written"])
            self.assertEqual(parse_incar(run / "INCAR")["MAGMOM"], NIO_MAGMOM)


class RunStatusTests(unittest.TestCase):
    PLAN = {"requested": "neural-paw", "applied": True, "stage": "launch"}

    def test_not_requested_skipped_and_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            self.assertEqual(density_init_run_status(run)["density_init_status"], "NOT_REQUESTED")
            skipped = density_init_run_status(run, {"requested": "neural-paw", "applied": False, "reason": "ISTART=1"})
            self.assertEqual(skipped["density_init_status"], "SKIPPED")
            self.assertFalse(skipped["density_init_executed"])
            pending = density_init_run_status(run, self.PLAN)
            self.assertEqual(pending["density_init_requested"], "neural-paw")
            self.assertEqual(pending["density_init_status"], "PENDING")
            self.assertIsNone(pending["density_init_compatible"])

    def test_promoted_run_is_compatible_and_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            initialize_density(run, backend=_backend(), grid=GRID)
            status = density_init_run_status(run, self.PLAN)
            self.assertIs(status["density_init_compatible"], True)
            self.assertIs(status["density_init_executed"], True)
            self.assertEqual(status["density_init_status"], "PROMOTED")
            self.assertFalse(status["density_init_fallback_occurred"])
            (run / FALLBACK_NAME).write_text(json.dumps({"action": "standard", "exit_code": 1}))
            self.assertFalse(density_init_run_status(run, self.PLAN)["density_init_executed"])

    def test_unsupported_potcar_schema_is_reported_as_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run", potcar_symbols=("Ni", "O"))
            (run / "density_init.json").write_text(
                json.dumps(
                    {
                        "format": "interfaceforge-density-init",
                        "backend": "neural-paw",
                        "status": "FAILED",
                        "failure_code": "UNSUPPORTED_POTCAR_SCHEMA",
                        "error": "InferenceError: POTCAR incompatible with the models",
                    }
                )
            )
            (run / FALLBACK_NAME).write_text(json.dumps({"action": "standard", "exit_code": 1}))
            status = density_init_run_status(run, self.PLAN)
            self.assertEqual(status["density_init_status"], "UNSUPPORTED_POTCAR_SCHEMA")
            self.assertIs(status["density_init_compatible"], False)
            self.assertIs(status["density_init_executed"], False)
            self.assertTrue(status["density_init_fallback_occurred"])
            self.assertEqual(status["density_init_fallback_action"], "standard")

    def test_hook_failure_before_any_report_is_failed_with_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            (run / "precondition").mkdir()
            (run / "precondition" / FALLBACK_NAME).write_text(json.dumps({"action": "standard", "exit_code": 127}))
            status = density_init_run_status(run, self.PLAN)
            self.assertEqual(status["density_init_status"], "FAILED")
            self.assertTrue(status["density_init_fallback_occurred"])
            self.assertIn("127", status["density_init_failure"])


def _opt_tree(root: Path, *, potcar_symbols: tuple[str, ...] = ("Ni_pv", "O")) -> Path:
    opt = root / "OPT"
    run = opt / "NiO"
    run.mkdir(parents=True)
    (opt / "KPOINTS").write_text("Gamma\n0\nGamma\n2 2 1\n0 0 0\n")
    (opt / "runvasp.sh").write_text("#!/bin/bash\nsrun vasp_std > vasp.out\n")
    (opt / "runvasp.sh").chmod(0o755)
    (run / "POTCAR").write_text(potcar(*potcar_symbols))
    (run / "INCAR").write_text(
        "ENCUT = 520\nPREC = Accurate\nISPIN = 2\nMAGMOM = 2*2.0 3*-2.0\nLDAU = .TRUE.\nLDAUL = 2 -1\n"
        "LDAUU = 4.6 0.0\nLDAUJ = 0.0 0.0\nLMAXMIX = 4\nIBRION = 2\nISIF = 2\nNSW = 200\n"
    )
    coords = "\n".join(f"0.1 0.1 {0.30 + i * 0.02:.4f}" for i in range(5))
    (run / "CONTCAR").write_text(f"opt\n1.0\n10 0 0\n0 10 0\n0 0 40\nNi O\n3 2\nDirect\n{coords}\n")
    return opt


class Step1PotcarAndStatusTests(unittest.TestCase):
    def test_step1_records_potcar_provenance_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepare_step1_series(_opt_tree(Path(tmp)), fresh_start=True)
            manifest = json.loads((Path(tmp) / "Step1" / "step1_manifest.json").read_text())
            row = manifest["runs"][0]["potcar"]
            self.assertEqual(row["potcar_variants"], {"Ni": "Ni_pv", "O": "O"})
            self.assertEqual(row["potcar_sha256"], _sha(Path(tmp) / "Step1" / "NiO" / "POTCAR"))
            self.assertIsNone(manifest["potcar_declaration"]["potcar_definitions"])

    def test_declared_mp_mapping_is_checked_and_passed_to_the_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production, mp = _defs(Path(tmp))
            with self.assertRaisesRegex(SafetyError, "declared 'Ni', POTCAR has 'Ni_pv'"):
                prepare_step1_series(_opt_tree(Path(tmp)), fresh_start=True, potcar_definitions=production)
            self.assertFalse((Path(tmp) / "Step1").exists())
            result = prepare_step1_series(
                Path(tmp) / "OPT",
                fresh_start=True,
                density_init="neural-paw",
                potcar_definitions=mp,
                potcar_generator="/home/user/bin/POTCAR_gen",
            )
            self.assertEqual(result["audit"]["status"], "PASS")
            launcher = (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text()
            self.assertIn(f"--potcar-definitions {mp.resolve()}", launcher)
            self.assertIn("--potcar-generator /home/user/bin/POTCAR_gen", launcher)
            manifest = json.loads((Path(tmp) / "Step1" / "step1_manifest.json").read_text())
            self.assertTrue(manifest["runs"][0]["potcar"]["consistent_with_definitions"])

    def test_precondition_with_abort_is_now_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_step1_series(
                _opt_tree(Path(tmp)),
                conservative=True,
                precondition=True,
                ramp_from=100.0,
                density_init="neural-paw",
                density_init_options={"on_failure": "abort"},
            )
            self.assertEqual(result["audit"]["status"], "PASS")
            launcher = (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text()
            self.assertIn("exit 3", launcher)

    def test_step1_status_reports_requested_vs_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepare_step1_series(_opt_tree(Path(tmp)), fresh_start=True, density_init="neural-paw")
            root = Path(tmp) / "Step1"
            pending = step1_status(root)
            self.assertEqual(pending["runs"][0]["density_init_status"], "PENDING")
            self.assertEqual(pending["density_init_tally"], {"PENDING": 1})
            initialize_density(root / "NiO", backend=_backend(), grid=(40, 40, 160))
            row = step1_status(root)["runs"][0]
            self.assertEqual(row["density_init_requested"], "neural-paw")
            self.assertTrue(row["density_init_compatible"])
            self.assertTrue(row["density_init_executed"])
            self.assertEqual(row["density_init_status"], "PROMOTED")
            plain = Path(tmp) / "plain"
            shutil.copytree(Path(tmp) / "OPT", plain / "OPT")
            prepare_step1_series(plain / "OPT", fresh_start=True)
            self.assertEqual(step1_status(plain / "Step1")["runs"][0]["density_init_status"], "NOT_REQUESTED")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("bash"), "bash required")
class LauncherFallbackExecutionTests(unittest.TestCase):
    """Run the wrapped launchers in bash with a stand-in hook and VASP."""

    def _run(self, *, precondition: bool, on_failure: str, hook: str, set_e: bool = False) -> tuple[Path, int]:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        run = Path(self._tmp.name)
        for name in ("POSCAR", "KPOINTS", "INCAR.precondition", "POTCAR"):
            (run / name).write_text("x\n")
        launcher = "#!/bin/bash\n" + ("set -e\n" if set_e else "") + "vasp_std > vasp.out\n"
        if precondition:
            launcher = wrap_launcher_with_precondition(launcher, launcher_name="runvasp.sh")
        launcher = wrap_launcher_with_density_init(
            launcher, launcher_name="runvasp.sh", hook=hook, on_failure=on_failure
        ).replace("vasp_std > vasp.out", "echo ran >> vasp.calls")
        (run / "runvasp.sh").write_text(launcher)
        completed = subprocess.run(["bash", "runvasp.sh"], cwd=run, capture_output=True, text=True, check=False)
        return run, completed.returncode

    def test_standard_fallback_is_recorded_and_vasp_still_runs(self) -> None:
        run, code = self._run(precondition=False, on_failure="standard", hook="false", set_e=True)
        self.assertEqual(code, 0)
        record = json.loads((run / FALLBACK_NAME).read_text())
        self.assertEqual((record["format"], record["action"], record["exit_code"]),
                         ("interfaceforge-density-init-fallback", "standard", 1))
        self.assertTrue((run / "vasp.calls").is_file())

    def test_abort_under_precondition_stops_the_job_before_any_vasp(self) -> None:
        run, code = self._run(precondition=True, on_failure="abort", hook="false")
        self.assertEqual(code, 3)
        self.assertEqual(json.loads((run / "precondition" / FALLBACK_NAME).read_text())["action"], "abort")
        self.assertEqual(list(run.rglob("vasp.calls")), [])

    def test_successful_hook_clears_a_stale_record(self) -> None:
        run, code = self._run(precondition=False, on_failure="abort", hook="true")
        self.assertEqual(code, 0)
        self.assertFalse((run / FALLBACK_NAME).exists())
        (run / FALLBACK_NAME).write_text("{}")
        subprocess.run(["bash", "runvasp.sh"], cwd=run, check=True, capture_output=True)
        self.assertFalse((run / FALLBACK_NAME).exists())


class BenchmarkSamePotcarTests(unittest.TestCase):
    def _pilot(self, root: Path, *, defs: Path | None = None) -> Path:
        sto = write_run(
            root / "src" / "sto", poscar=STO_POSCAR, incar=STO_INCAR + "IBRION = 2\nNSW = 50\n",
            potcar_symbols=("Sr_sv", "Ti_pv", "O"),
        )
        (sto / "CONTCAR").write_text(STO_POSCAR)
        (sto / "runvasp.sh").write_text("#!/bin/bash\nsrun vasp_std > vasp.out\n")
        case = {"name": "sto", "source": "src/sto", "grid": [36, 36, 36]}
        if defs is not None:
            case["potcar_definitions"] = str(defs)
        (root / "pilot.yaml").write_text(yaml.safe_dump({"cases": [case]}))
        return root / "pilot.yaml"

    def test_potcar_provenance_is_recorded_and_declaration_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            production, mp = _defs(root)
            with self.assertRaisesRegex(SafetyError, "declared 'Ti_sv', POTCAR has 'Ti_pv'"):
                prepare_benchmark(self._pilot(root, defs=production), root / "bench")
            shutil.rmtree(root / "bench", ignore_errors=True)
            shutil.rmtree(root / "src")
            prepare_benchmark(self._pilot(root, defs=mp), root / "bench")
            manifest = json.loads((root / "bench" / "density_init_bench.json").read_text())
            case = manifest["cases"][0]
            self.assertEqual(case["potcar"]["potcar_variants"], {"Sr": "Sr_sv", "Ti": "Ti_pv", "O": "O"})
            self.assertEqual(case["shared_inputs_sha256"]["POTCAR"]["standard"], case["potcar"]["potcar_sha256"])
            self.assertIn("--potcar-definitions", (root / "bench" / "sto" / "neural" / "runvasp.sh").read_text())

    def test_replaced_potcar_in_one_arm_blocks_the_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_benchmark(self._pilot(root), root / "bench")
            (root / "bench" / "sto" / "neural" / "POTCAR").write_text(potcar("Sr_sv", "Ti_pv", "O") + "\n")
            report = compare_benchmark(root / "bench", write=False)
            case = report["cases"][0]
            self.assertEqual(case["verdict"], "INPUT_MISMATCH")
            self.assertFalse(case["checks"]["same_inputs"])
            self.assertEqual(report["summary"]["input_mismatch"], 1)
            self.assertIn("not evaluated", str(report["summary"]["acceptance"]["1_same_electronic_solution"]))

    def test_neural_arm_fallback_is_a_neural_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_benchmark(self._pilot(root), root / "bench")
            neural = root / "bench" / "sto" / "neural"
            (neural / "density_init.json").write_text(
                json.dumps({"format": "interfaceforge-density-init", "status": "PROMOTED", "active": True})
            )
            (neural / FALLBACK_NAME).write_text(json.dumps({"action": "standard", "exit_code": 1}))
            finish_run(root / "bench" / "sto" / "standard", scf=22, energy=-38.5, elapsed=600.0)
            finish_run(neural, scf=22, energy=-38.5, elapsed=600.0)
            case = compare_benchmark(root / "bench", write=False)["cases"][0]
            self.assertTrue(case["neural_fallback_occurred"])
            self.assertIs(case["checks"]["neural_init_active"], False)
            self.assertEqual(case["verdict"], "NEURAL_FAILED")


if __name__ == "__main__":
    unittest.main()
