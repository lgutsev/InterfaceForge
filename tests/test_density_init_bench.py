"""Output parsing, magnetic audit, launcher hooks, Step1 integration and the benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from density_init_fixtures import (
    NIO_INCAR,
    NIO_MAGMOM,
    STO_INCAR,
    STO_POSCAR,
    FakeRunner,
    write_run,
)

from interfaceforge.density_init import NeuralPawInitializer, initialize_density
from interfaceforge.density_init.benchmark import compare_benchmark, prepare_benchmark, render_case_table
from interfaceforge.density_init.launch import DENSITY_INIT_MARKER, wrap_launcher_with_density_init
from interfaceforge.density_init.outputs import (
    audit_initialized_run,
    magnetic_pattern,
    parse_oszicar,
    parse_outcar,
)
from interfaceforge.errors import SafetyError
from interfaceforge.step1_launch import launch_step1_runs
from interfaceforge.vasp import parse_incar, prepare_step1_series, wrap_launcher_with_precondition


def outcar_text(
    *,
    energy: float,
    forces: list[list[float]],
    stress: list[float],
    moments: list[float] | None,
    elapsed: float,
    nelm: int = 60,
    converged: bool = True,
    grid: tuple[int, int, int] = (24, 24, 24),
) -> str:
    lines = [
        " vasp.6.4.2 18Apr23 (build Jun 01 2023) complex",
        f" dimension x,y,z NGXF=    {grid[0]} NGYF=    {grid[1]} NGZF=    {grid[2]}",
        f"   NELM   =     {nelm};   NELMIN=  2; NELMDL= -5     # of ELM steps",
        "------------------------ aborting loop because EDIFF is reached ----------------------------------------"
        if converged
        else "------------------------ aborting loop EDIFF was not reached (unconverged)  ----------------------------",
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        f"  free  energy   TOTEN  =      {energy - 0.001:.8f} eV",
        f"  energy  without entropy=     {energy:.8f}  energy(sigma->0) =     {energy:.8f}",
        "  FORCE on cell =-STRESS in cart. coord.  units (eV):",
        "  in kB " + " ".join(f"{value:11.5f}" for value in stress),
        "",
        " POSITION                                       TOTAL-FORCE (eV/Angst)",
        " -----------------------------------------------------------------------------------",
    ]
    lines += [f"      0.00000      0.00000      0.00000     {f[0]:10.6f} {f[1]:10.6f} {f[2]:10.6f}" for f in forces]
    lines.append(" -----------------------------------------------------------------------------------")
    if moments is not None:
        lines += [" magnetization (x)", "", "# of ion       s       p       d       tot", "-" * 42]
        lines += [f"    {i + 1}        0.000   0.000  {m:6.3f}  {m:6.3f}" for i, m in enumerate(moments)]
        lines += ["-" * 42, f"tot          0.000   0.000   0.000  {sum(moments):6.3f}"]
    lines += [
        " General timing and accounting informations for this job:",
        f"                            Elapsed time (sec):     {elapsed:.3f}",
    ]
    return "\n".join(lines) + "\n"


def oszicar_text(scf_steps: int, energy: float, mag: float | None = None) -> str:
    lines = ["       N       E                     dE             d eps       ncg     rms          rms(c)"]
    lines += [f"DAV:  {i + 1:3d}    -0.1E+03   -0.1E+01   -0.1E+00  1000   0.1E+00" for i in range(scf_steps)]
    mag_part = f"  mag=     {mag:.4f}" if mag is not None else ""
    lines.append(f"   1 F= {energy:.8E} E0= {energy:.8E}  d E =-.1E+00{mag_part}")
    return "\n".join(lines) + "\n"


FORCES = [[0.01, 0.0, -0.02], [-0.01, 0.0, 0.02], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
STRESS = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def finish_run(run: Path, *, scf: int, energy: float, elapsed: float, moments=None, forces=FORCES, stress=STRESS):
    (run / "OUTCAR").write_text(
        outcar_text(energy=energy, forces=forces, stress=stress, moments=moments, elapsed=elapsed)
    )
    (run / "OSZICAR").write_text(oszicar_text(scf, energy, sum(moments) if moments else None))


class OutputParserTests(unittest.TestCase):
    def test_outcar_and_oszicar_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            finish_run(run, scf=17, energy=-40.123456, elapsed=321.5, moments=[1.68, -1.68, 0.0, 0.0])
            outcar = parse_outcar(run / "OUTCAR")
            self.assertAlmostEqual(outcar["energy_sigma0_ev"], -40.123456)
            self.assertEqual(outcar["forces"], FORCES)
            self.assertEqual(outcar["stress_kb"], STRESS)
            self.assertEqual(outcar["local_moments"], [1.68, -1.68, 0.0, 0.0])
            self.assertEqual(outcar["elapsed_s"], 321.5)
            self.assertEqual(outcar["nelm"], 60)
            self.assertEqual(parse_oszicar(run / "OSZICAR")["scf_steps_per_ionic"], [17])

    def test_magnetic_pattern_states(self) -> None:
        reference = [1.7, -1.7, 0.0, 0.0]
        self.assertEqual(magnetic_pattern([1.65, -1.66, 0.0, 0.0], reference)["status"], "PRESERVED")
        self.assertEqual(magnetic_pattern([-1.65, 1.66, 0.0, 0.0], reference)["status"], "PRESERVED_GLOBAL_FLIP")
        broken = magnetic_pattern([1.65, 1.66, 0.0, 0.0], reference)
        self.assertEqual(broken["status"], "BROKEN")
        self.assertEqual(broken["mismatched_sites"], [2])
        quenched = magnetic_pattern([1.65, -0.1, 0.0, 0.0], reference)
        self.assertEqual(quenched["status"], "BROKEN")
        self.assertEqual(quenched["quenched_sites"], [2])
        self.assertEqual(magnetic_pattern(None, reference)["status"], "UNKNOWN")
        self.assertEqual(magnetic_pattern([0.0] * 4, None)["status"], "NOT_APPLICABLE")

    def test_density_init_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            self.assertEqual(audit_initialized_run(run, write=False)["status"], "INCOMPLETE")
            finish_run(run, scf=20, energy=-40.0, elapsed=10.0, moments=[1.7, -1.69, 0.0, 0.0])
            payload = audit_initialized_run(run)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["magnetic_pattern"]["status"], "PRESERVED")
            self.assertTrue((run / "density_init_audit.json").is_file())
            finish_run(run, scf=20, energy=-40.0, elapsed=10.0, moments=[1.7, 1.69, 0.0, 0.0])
            self.assertEqual(audit_initialized_run(run)["status"], "FAIL")


class LauncherHookTests(unittest.TestCase):
    LAUNCHER = "#!/bin/bash\n#SBATCH -N 1\nmodule load vasp\nsrun vasp_std > vasp.out\n"

    def test_plain_launcher_hook_runs_before_vasp_with_fallback(self) -> None:
        wrapped = wrap_launcher_with_density_init(self.LAUNCHER, launcher_name="runvasp.sh", hook="iface-init")
        lines = wrapped.splitlines()
        self.assertIn(DENSITY_INIT_MARKER, wrapped)
        self.assertLess(lines.index("if ! iface-init; then"), lines.index("srun vasp_std > vasp.out"))
        self.assertIn("standard start", wrapped)
        self.assertEqual(wrap_launcher_with_density_init(wrapped, launcher_name="runvasp.sh", hook="x"), wrapped)
        aborting = wrap_launcher_with_density_init(
            self.LAUNCHER, launcher_name="runvasp.sh", hook="iface-init", on_failure="abort"
        )
        self.assertIn("exit 3", aborting)

    def test_preconditioned_launcher_seeds_the_static_scf(self) -> None:
        pre = wrap_launcher_with_precondition(self.LAUNCHER, launcher_name="runvasp.sh")
        wrapped = wrap_launcher_with_density_init(pre, launcher_name="runvasp.sh", hook="iface-init")
        self.assertIn("mv -f INCAR.precondition INCAR && { iface-init ||", wrapped)
        with self.assertRaises(SafetyError):
            wrap_launcher_with_density_init(pre, launcher_name="runvasp.sh", hook="x", on_failure="abort")

    def test_ambiguous_launcher_refused(self) -> None:
        with self.assertRaises(SafetyError):
            wrap_launcher_with_density_init(
                self.LAUNCHER + "srun vasp_gam\n", launcher_name="runvasp.sh", hook="iface-init"
            )


def _opt_tree(root: Path, *, wavecar: bool, magmom: str = "2*2.0 3*-2.0") -> Path:
    opt = root / "OPT"
    run = opt / "NiO"
    run.mkdir(parents=True)
    (opt / "KPOINTS").write_text("Gamma\n0\nGamma\n2 2 1\n0 0 0\n")
    launcher = opt / "runvasp.sh"
    launcher.write_text("#!/bin/bash\nsrun vasp_std > vasp.out\n")
    launcher.chmod(0o755)
    (run / "INCAR").write_text(
        f"ENCUT = 520\nPREC = Accurate\nISPIN = 2\nMAGMOM = {magmom}\nLDAU = .TRUE.\nLDAUL = 2 -1\n"
        "LDAUU = 4.6 0.0\nLDAUJ = 0.0 0.0\nLMAXMIX = 4\nIBRION = 2\nISIF = 2\nNSW = 200\n"
    )
    coords = "\n".join(f"0.1 0.1 {0.30 + i * 0.02:.4f}" for i in range(5))
    (run / "CONTCAR").write_text(f"opt\n1.0\n10 0 0\n0 10 0\n0 0 40\nNi O\n3 2\nDirect\n{coords}\n")
    if wavecar:
        (run / "WAVECAR").write_text("x" * 4096)
    return opt


class Step1IntegrationTests(unittest.TestCase):
    def test_default_is_standard_and_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=False)
            result = prepare_step1_series(opt, fresh_start=True)
            self.assertEqual(result["density_init"], {"backend": "standard"})
            self.assertNotIn(DENSITY_INIT_MARKER, (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text())

    def test_fresh_start_run_gets_launch_hook_and_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=False)
            result = prepare_step1_series(
                opt,
                fresh_start=True,
                density_init="neural-paw",
                density_init_options={"backend_options": {"python": "/envs/ndi/bin/python"}},
            )
            self.assertEqual(result["audit"]["status"], "PASS")
            self.assertEqual(result["density_init"]["applied_runs"], 1)
            launcher = (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text()
            self.assertIn("initialize-density . --backend neural-paw --magmom-source incar", launcher)
            self.assertIn("--ndi-python /envs/ndi/bin/python", launcher)
            # Step1 renders ENCUT 400/PREC Normal, so the OPT grid is not reusable -> dry run
            self.assertIn("--grid-dry-run-command 'srun vasp_std > vasp.out'", launcher)
            incar = parse_incar(Path(tmp) / "Step1" / "NiO" / "INCAR")
            self.assertEqual(incar["MAGMOM"], "2*2.0 3*-2.0")
            self.assertNotIn("ICHARG", incar)  # set only by the job, after a successful seed
            manifest = json.loads((Path(tmp) / "Step1" / "step1_manifest.json").read_text())
            self.assertTrue(manifest["runs"][0]["density_init"]["applied"])
            self.assertEqual(manifest["density_init"]["backend"], "neural-paw")

    def test_wavecar_restart_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=True)
            result = prepare_step1_series(opt, density_init="neural-paw")
            self.assertEqual(result["density_init"]["skipped_runs"], 1)
            self.assertNotIn(DENSITY_INIT_MARKER, (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text())

    def test_preconditioned_nio_seeds_the_preconditioner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=False)
            result = prepare_step1_series(
                opt, conservative=True, precondition=True, ramp_from=100.0, density_init="neural-paw"
            )
            self.assertEqual(result["audit"]["status"], "PASS")
            launcher = (Path(tmp) / "Step1" / "NiO" / "runvasp.sh").read_text()
            self.assertIn("mv -f INCAR.precondition INCAR && {", launcher)

    def test_initializer_moments_for_afm_rejected_at_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=False)
            with self.assertRaisesRegex(SafetyError, "mixed signs"):
                prepare_step1_series(
                    opt,
                    fresh_start=True,
                    density_init="neural-paw",
                    density_init_options={"magmom_source": "initializer"},
                )
            self.assertFalse((Path(tmp) / "Step1").exists())

    def test_step1_launch_accepts_only_recorded_icharg_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opt = _opt_tree(Path(tmp), wavecar=False)
            prepare_step1_series(opt, fresh_start=True)
            run = Path(tmp) / "Step1" / "NiO"
            from density_init_fixtures import potcar

            (run / "POTCAR").write_text(potcar("Ni_pv", "O"))
            initialize_density(run, backend=NeuralPawInitializer(runner=FakeRunner()), grid=(40, 40, 160))
            plan = launch_step1_runs([Path(tmp) / "Step1"])
            self.assertEqual(plan["planned"][0]["kind"], "prepared+density-init")
            (run / "INCAR").write_text((run / "INCAR").read_text() + "NELM = 500\n")
            with self.assertRaisesRegex(SafetyError, "changed since step1-prepare"):
                launch_step1_runs([Path(tmp) / "Step1"])


class BenchmarkTests(unittest.TestCase):
    def _sources(self, root: Path) -> Path:
        sto = write_run(
            root / "src" / "sto", poscar=STO_POSCAR, incar=STO_INCAR + "IBRION = 2\nNSW = 50\n",
            potcar_symbols=("Sr_sv", "Ti_pv", "O"),
        )
        (sto / "CONTCAR").write_text(STO_POSCAR)
        (sto / "OUTCAR").write_text(" dimension x,y,z NGXF=    36 NGYF=    36 NGZF=    36\n")
        nio = write_run(root / "src" / "nio", incar=NIO_INCAR + "IBRION = 0\nNSW = 400\nISIF = 0\n")
        (nio / "CONTCAR").write_text((nio / "POSCAR").read_text())
        for source in (sto, nio):
            (source / "runvasp.sh").write_text("#!/bin/bash\nsrun vasp_std > vasp.out\n")
        pilot = root / "pilot.yaml"
        pilot.write_text(
            yaml.safe_dump(
                {
                    "cases": [
                        {"name": "sto", "source": "src/sto", "category": "nonmagnetic-perovskite"},
                        {
                            "name": "nio_afm2",
                            "source": "src/nio",
                            "category": "NiO AFM-II",
                            "expect_magnetic_order": "afm-ii",
                            "grid": [24, 24, 24],
                        },
                    ]
                }
            )
        )
        return pilot

    def test_prepare_creates_identical_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = self._sources(root)
            dry = prepare_benchmark(pilot, root / "bench", dry_run=True)
            self.assertEqual([case["magnetic_order"] for case in dry["cases"]], ["non-spin-polarized", "mixed-sign"])
            self.assertFalse((root / "bench").exists())
            result = prepare_benchmark(pilot, root / "bench", backend_options={"python": "/envs/ndi/bin/python"})
            self.assertEqual(result["cases"], 2)
            for case in ("sto", "nio_afm2"):
                std, neu = root / "bench" / case / "standard", root / "bench" / case / "neural"
                for name in ("POSCAR", "POTCAR", "KPOINTS"):
                    self.assertEqual((std / name).read_bytes(), (neu / name).read_bytes())
                std_tags, neu_tags = parse_incar(std / "INCAR"), parse_incar(neu / "INCAR")
                self.assertEqual(std_tags.pop("ICHARG"), "2")
                self.assertEqual(std_tags, neu_tags)
                self.assertEqual(std_tags["NSW"], "0")
                self.assertEqual(json.loads((std / "density_init.json").read_text())["status"], "STANDARD_START")
                self.assertIn(DENSITY_INIT_MARKER, (neu / "runvasp.sh").read_text())
                self.assertIn("exit 3", (neu / "runvasp.sh").read_text())  # failures are counted, not hidden
            self.assertEqual(parse_incar(root / "bench" / "nio_afm2" / "neural" / "INCAR")["MAGMOM"], NIO_MAGMOM)
            self.assertEqual(parse_incar(root / "bench" / "nio_afm2" / "neural" / "INCAR")["ISIF"], "2")
            self.assertIn("--grid 36 36 36", (root / "bench" / "sto" / "neural" / "runvasp.sh").read_text())
            with self.assertRaisesRegex(SafetyError, "overwrite"):
                prepare_benchmark(pilot, root / "bench")

    def test_afm_case_requires_signed_magmom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = self._sources(root)
            nio = root / "src" / "nio" / "INCAR"
            nio.write_text(nio.read_text().replace(NIO_MAGMOM, "2*1.7 2*0"))
            with self.assertRaisesRegex(SafetyError, "signed AFM-II"):
                prepare_benchmark(pilot, root / "bench", dry_run=True)

    def _finish(self, bench: Path, *, neural_moments, neural_energy_shift=0.0) -> None:
        sto_std, sto_neu = bench / "sto" / "standard", bench / "sto" / "neural"
        finish_run(sto_std, scf=22, energy=-38.5, elapsed=600.0)
        finish_run(sto_neu, scf=15, energy=-38.5 + 1e-6, elapsed=420.0)
        nio_std, nio_neu = bench / "nio_afm2" / "standard", bench / "nio_afm2" / "neural"
        finish_run(nio_std, scf=40, energy=-40.0, elapsed=1000.0, moments=[1.68, -1.68, 0.0, 0.0])
        finish_run(nio_neu, scf=35, energy=-40.0 + neural_energy_shift, elapsed=900.0, moments=neural_moments)
        for neu in (sto_neu, nio_neu):
            (neu / "density_init.json").write_text(
                json.dumps(
                    {
                        "format": "interfaceforge-density-init",
                        "status": "PROMOTED",
                        "active": True,
                        "backend": "neural-paw",
                        "timing": {"overhead_s": 30.0, "inference_s": 5.0, "model_load_s": 20.0},
                    }
                )
            )

    def test_compare_reports_same_solution_and_separates_speedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_benchmark(self._sources(root), root / "bench")
            incomplete = compare_benchmark(root / "bench", write=False)
            self.assertEqual(incomplete["summary"]["finished"], 0)
            self.assertIn("not evaluated", str(incomplete["summary"]["acceptance"]["5_scf_reduced_in_some_cases"]))
            self._finish(root / "bench", neural_moments=[1.68, -1.679, 0.0, 0.0])
            report = compare_benchmark(root / "bench")
            cases = {case["name"]: case for case in report["cases"]}
            sto = cases["sto"]
            self.assertEqual(sto["verdict"], "SAME_SOLUTION")
            self.assertEqual(sto["scf_iterations"], {"standard": 22, "neural": 15, "difference": -7})
            self.assertEqual(sto["inference_overhead_s"], {"standard": 0.0, "neural": 30.0})
            self.assertEqual(sto["total_wall_s"]["neural"], 450.0)
            self.assertAlmostEqual(sto["scf_speedup"], 22 / 15)
            self.assertAlmostEqual(sto["end_to_end_speedup"], 600 / 450)
            nio = cases["nio_afm2"]
            self.assertEqual(nio["verdict"], "SAME_SOLUTION")
            self.assertEqual(nio["magnetic_pattern"]["neural"]["status"], "PRESERVED")
            summary = report["summary"]
            self.assertTrue(summary["acceptance"]["4_afm_ii_preserved"])
            self.assertTrue(summary["acceptance"]["6_end_to_end_wall_time_saved"])
            self.assertEqual(summary["failures"], {"standard": 0, "neural": 0})
            table = render_case_table(nio)
            self.assertIn("| SCF iterations | 40 | 35 | -5 |", table)
            self.assertIn("| Inference time (s) | 0 | 30.000 | 30.000 |", table)
            self.assertIn("PRESERVED", table)
            for suffix in ("json", "md", "tsv"):
                self.assertTrue((root / "bench" / f"density_init_bench_report.{suffix}").is_file())

    def test_compare_flags_broken_afm_and_energy_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_benchmark(self._sources(root), root / "bench")
            self._finish(root / "bench", neural_moments=[1.2, 1.2, 0.0, 0.0], neural_energy_shift=0.3)
            report = compare_benchmark(root / "bench", write=False)
            nio = next(case for case in report["cases"] if case["name"] == "nio_afm2")
            self.assertEqual(nio["verdict"], "DIFFERENT_SOLUTION")
            self.assertEqual(nio["magnetic_pattern"]["neural"]["status"], "BROKEN")
            self.assertFalse(report["summary"]["acceptance"]["4_afm_ii_preserved"])
            self.assertFalse(report["summary"]["acceptance"]["1_same_electronic_solution"])


if __name__ == "__main__":
    unittest.main()
