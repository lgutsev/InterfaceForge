"""Tests for the refined Step1 stability diagnostic (``diagnose_step1_run``).

The diagnostic separates hard instability (automatic repair) from
review-level warnings.  The central regression is the real false positive: a
completed 400/400, ~302 K, clean-tailed NiO run whose ionic step 1 sat ~77 eV
from the free-energy reference (expected magnetic DFT+U startup relaxation)
was marked unstable.  Every trajectory here is synthetic (tests/step1_fixtures).
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from interfaceforge.errors import SafetyError
from interfaceforge.step1_repair import (
    DEFAULT_CATASTROPHIC_ENERGY_EV,
    DEFAULT_ENERGY_JUMP_EV,
    DEFAULT_REFERENCE_WINDOW_STEPS,
    DEFAULT_STARTUP_GRACE_STEPS,
    SCF_HARD_FRACTION,
    SCF_WARN_FRACTION,
    apply_precondition,
    diagnose_step1_run,
    parse_step1_oszicar,
    precondition_blocker,
)
from interfaceforge.vasp import parse_incar, wrap_launcher_with_precondition

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from step1_fixtures import PLAIN_LAUNCHER, linear_ramp, tree_snapshot, write_step1_run  # noqa: E402

OLD_KEYS = {
    "run",
    "md_steps",
    "last_step",
    "reference_free_energy_ev",
    "energy_jump_limit_ev",
    "temperature_limit_k",
    "first_bad_step",
    "first_bad_reasons",
    "scf_nelm",
    "scf_ceiling_steps",
    "scf_window_steps",
    "scf_ceiling_fraction",
    "scf_unreliable",
    "unstable",
}
NEW_KEYS = {
    "severity",
    "trajectory_stable",
    "hard_reasons",
    "warnings",
    "warning_classes",
    "benign_warnings_only",
    "startup_grace_steps",
    "startup_excursion_settled",
    "startup_excursion_downhill",
    "reference_window",
    "reference_source",
    "catastrophic_energy_limit_ev",
    "temperature_warning_k",
    "startup_excursion_steps",
    "startup_max_excursion_ev",
    "isolated_spike_steps",
    "max_local_jump_ev",
    "max_local_jump_step",
    "corroborated",
    "torn_final_line",
}

REFERENCE_F = -10.0


def baseline_energy(step: int) -> float:
    """A quiet NiO-like trajectory: F wanders within 0.2 eV of the reference."""

    return REFERENCE_F + 0.05 * ((step * 7) % 5)


def warm_temperature(step: int) -> float:
    """~302 K with a few K of thermostat noise (well below every warning level)."""

    return 302.0 + 4.0 * math.sin(step)


def energies_with(offsets: dict[int, float | None]) -> Callable[[int], float | None]:
    """Baseline energies with ``{step: offset_from_reference}`` overrides (None -> ``******``)."""

    def energy(step: int) -> float | None:
        if step in offsets:
            offset = offsets[step]
            return None if offset is None else REFERENCE_F + offset
        return baseline_energy(step)

    return energy


def md_line(step: int, energy: float, temperature: float = 301.4) -> str:
    """One complete OSZICAR MD line in the layout of ``step1_fixtures.oszicar_text`` (no newline)."""

    return (
        f"{step:5d} T= {temperature:7.1f} E= {energy + 1.0:.8E} F= {energy:.8E} E0= {energy:.8E} "
        "EK= 0.10000E+01 SP= 0.00E+00 SK= 0.00E+00"
    )


def temperatures_with(overrides: dict[int, float | None]) -> Callable[[int], float | None]:
    def temperature(step: int) -> float | None:
        return overrides[step] if step in overrides else warm_temperature(step)

    return temperature


class Step1DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._count = 0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_dir(self, **kwargs: Any) -> Path:
        """A completed 400/400 run at ~302 K with a clean trajectory unless overridden."""

        self._count += 1
        options: dict[str, Any] = {
            "nsw": 400,
            "steps": 400,
            "outcar": "finished",
            "temperatures": warm_temperature,
            "energies": baseline_energy,
        }
        options.update(kwargs)
        return write_step1_run(self.root, f"run{self._count}", **options)

    def diagnose(self, run: Path, **kwargs: Any) -> dict[str, Any]:
        row = diagnose_step1_run(run, **kwargs)
        self.assertLessEqual(OLD_KEYS | NEW_KEYS, set(row))
        self.assertEqual(row["trajectory_stable"], not row["unstable"])
        self.assertEqual(row["severity"] == "unstable", row["unstable"])
        return row

    # ------------------------------------------------------------------ #
    # warnings that must stay warnings

    def test_clean_run_is_ok(self) -> None:
        row = self.diagnose(self.run_dir())
        self.assertEqual(row["severity"], "ok")
        self.assertFalse(row["unstable"])
        self.assertEqual(row["warning_classes"], [])
        self.assertEqual(row["warnings"], [])
        self.assertEqual(row["hard_reasons"], [])
        self.assertFalse(row["benign_warnings_only"])
        self.assertIsNone(row["first_bad_step"])
        self.assertEqual(row["first_bad_reasons"], [])
        self.assertEqual(row["reference_window"], [11, 20])
        self.assertEqual(row["reference_source"], "post_grace_window")
        self.assertIsNone(row["startup_excursion_settled"])
        self.assertIsNone(row["startup_excursion_downhill"])
        self.assertFalse(row["torn_final_line"])
        self.assertEqual(row["startup_grace_steps"], DEFAULT_STARTUP_GRACE_STEPS)
        self.assertEqual(row["catastrophic_energy_limit_ev"], DEFAULT_CATASTROPHIC_ENERGY_EV)
        self.assertEqual(row["energy_jump_limit_ev"], DEFAULT_ENERGY_JUMP_EV)
        self.assertEqual(row["temperature_limit_k"], 1200.0)
        self.assertEqual(row["temperature_warning_k"], 600.0)
        self.assertEqual(row["md_steps"], 400)
        self.assertEqual(row["last_step"], 400)
        self.assertEqual(row["scf_window_steps"], 390)  # post-grace rows only
        self.assertEqual(row["scf_ceiling_fraction"], 0.0)
        self.assertLess(row["max_local_jump_ev"], 1.0)

    def test_completed_run_with_step1_excursion_is_a_benign_warning(self) -> None:
        # Real false positive: 400/400, ~302 K, step 1 ~77 eV off, clean afterwards.
        run = self.run_dir(energies=energies_with({1: 77.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertTrue(row["trajectory_stable"])
        self.assertTrue(row["benign_warnings_only"])
        self.assertEqual(row["warning_classes"], ["startup_energy_excursion"])
        self.assertEqual(row["startup_excursion_steps"], [1])
        self.assertTrue(row["startup_excursion_settled"])
        self.assertTrue(row["startup_excursion_downhill"])
        self.assertAlmostEqual(row["startup_max_excursion_ev"], 77.0, delta=0.5)
        self.assertIsNone(row["first_bad_step"])
        self.assertEqual(row["first_bad_reasons"], [])
        self.assertEqual(row["hard_reasons"], [])
        self.assertEqual(row["corroborated"], [])
        self.assertEqual(len(row["warnings"]), 1)
        self.assertIn("startup energy excursion", row["warnings"][0])
        self.assertNotIn("not settled", row["warnings"][0])
        self.assertAlmostEqual(row["reference_free_energy_ev"], REFERENCE_F, delta=0.3)
        self.assertEqual(row["reference_window"], [11, 20])
        self.assertEqual(row["reference_source"], "post_grace_window")

    def test_multi_step_startup_excursion_is_still_a_warning(self) -> None:
        run = self.run_dir(energies=energies_with({1: 77.0, 2: 64.0, 3: 55.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertTrue(row["benign_warnings_only"])
        self.assertEqual(row["startup_excursion_steps"], [1, 2, 3])
        self.assertTrue(row["startup_excursion_settled"])
        self.assertIsNone(row["first_bad_step"])

    def test_isolated_post_grace_spike_is_a_warning(self) -> None:
        run = self.run_dir(energies=energies_with({150: 80.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertFalse(row["benign_warnings_only"])
        self.assertEqual(row["warning_classes"], ["isolated_energy_spike"])
        self.assertEqual(row["isolated_spike_steps"], [150])
        self.assertIsNone(row["first_bad_step"])
        # the jumps into (step 150) and out of (step 151) the spike are both ~80 eV
        self.assertIn(row["max_local_jump_step"], (150, 151))
        self.assertAlmostEqual(row["max_local_jump_ev"], 80.0, delta=0.5)

    def test_startup_excursion_with_elevated_scf_stays_a_warning(self) -> None:
        # W1 + W3 is the expected magnetic DFT+U fresh-start pattern: review, not repair.
        run = self.run_dir(
            energies=energies_with({1: 77.0}),
            scf_iterations=lambda step: 60 if step > 10 and step % 10 < 3 else 3,
        )
        row = self.diagnose(run)
        self.assertAlmostEqual(row["scf_ceiling_fraction"], 0.3)
        self.assertEqual(row["scf_ceiling_steps"], 117)
        self.assertEqual(row["scf_window_steps"], 390)
        self.assertFalse(row["scf_unreliable"])
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertFalse(row["benign_warnings_only"])
        self.assertEqual(row["warning_classes"], ["scf_elevated", "startup_energy_excursion"])
        self.assertEqual(row["corroborated"], [])
        self.assertIsNone(row["first_bad_step"])

    def test_elevated_temperature_alone_is_a_warning(self) -> None:
        run = self.run_dir(temperatures=temperatures_with({200: 700.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertEqual(row["warning_classes"], ["temperature_elevated"])
        self.assertFalse(row["benign_warnings_only"])
        self.assertIn("700 K", row["warnings"][0])

    def test_trajectory_shorter_than_grace_window_with_step1_excursion_is_a_warning(self) -> None:
        run = self.run_dir(steps=6, outcar="running", energies=energies_with({1: 77.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertTrue(row["benign_warnings_only"])
        self.assertEqual(row["startup_excursion_steps"], [1])
        self.assertTrue(row["startup_excursion_settled"])
        # no post-grace rows: the reference falls back to the first rows, no SCF window
        self.assertEqual(row["reference_window"], [1, 6])
        self.assertEqual(row["reference_source"], "first_rows")
        self.assertEqual(row["scf_window_steps"], 0)
        self.assertIsNone(row["scf_ceiling_fraction"])
        self.assertFalse(row["scf_unreliable"])
        self.assertIsNone(row["max_local_jump_ev"])

    # ------------------------------------------------------------------ #
    # startup excursions that are NOT benign

    def test_startup_excursion_still_departing_on_the_last_recorded_row_is_not_benign(self) -> None:
        # An interrupted 6-step run whose last row sits 77 eV off: the excursion has not
        # been shown to settle, so resume must not treat it as a benign transient.
        run = self.run_dir(steps=6, outcar="running", energies=energies_with({6: 77.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertEqual(row["warning_classes"], ["startup_energy_excursion"])
        self.assertEqual(row["startup_excursion_steps"], [6])
        self.assertIs(row["startup_excursion_settled"], False)
        self.assertFalse(row["benign_warnings_only"])
        self.assertIn("not settled", row["warnings"][0])
        self.assertIsNone(row["first_bad_step"])

        # 3 rows [+77, +77, relaxed]: too short to call it settled -> review, not benign
        run = self.run_dir(steps=3, outcar="running", energies=energies_with({1: 77.0, 2: 77.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertFalse(row["benign_warnings_only"])

    def test_short_trajectory_straddling_two_levels_is_not_reported_ok(self) -> None:
        # Steps 4-6 are still ~70 eV above steps 1-3: the first-rows median would sit
        # between the two levels (within 50 eV of both) and hide the excursion.
        run = self.run_dir(steps=6, outcar="running", energies=energies_with({4: 77.0, 5: 70.0, 6: 66.0}))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertFalse(row["benign_warnings_only"])
        self.assertEqual(row["reference_source"], "latest_rows")
        self.assertEqual(row["reference_window"], [4, 6])
        self.assertEqual(row["startup_excursion_steps"], [1, 2, 3])
        self.assertIs(row["startup_excursion_downhill"], False)

        # 2 rows, step 1 77 eV above step 2: a step-1 excursion that relaxed -> benign
        row = self.diagnose(self.run_dir(steps=2, outcar="running", energies=energies_with({1: 77.0})))
        self.assertEqual(row["severity"], "warning")
        self.assertEqual(row["reference_source"], "latest_rows")
        self.assertEqual(row["reference_window"], [2, 2])
        self.assertEqual(row["startup_excursion_steps"], [1])
        self.assertIs(row["startup_excursion_settled"], True)
        self.assertIs(row["startup_excursion_downhill"], True)
        self.assertTrue(row["benign_warnings_only"])

        # 2 rows, step 2 77 eV above step 1: the energy is rising on the last row -> not benign
        row = self.diagnose(self.run_dir(steps=2, outcar="running", energies=energies_with({2: 77.0})))
        self.assertEqual(row["severity"], "warning")
        self.assertIs(row["startup_excursion_downhill"], False)
        self.assertFalse(row["benign_warnings_only"])

    def test_energy_rise_after_startup_that_stays_up_is_not_benign(self) -> None:
        # F sits at the reference for steps 1-8, then rises by 80 eV at step 9 and stays
        # there: energy injection or a wrong electronic state, not a startup relaxation.
        run = self.run_dir(energies=lambda step: REFERENCE_F + (80.0 if step >= 9 else 0.0))
        row = self.diagnose(run)
        self.assertEqual(row["severity"], "warning")
        self.assertFalse(row["unstable"])
        self.assertEqual(row["warning_classes"], ["startup_energy_excursion"])
        self.assertEqual(row["startup_excursion_steps"], list(range(1, 9)))
        self.assertIs(row["startup_excursion_settled"], True)
        self.assertIs(row["startup_excursion_downhill"], False)
        self.assertFalse(row["benign_warnings_only"])
        self.assertIn("rose after startup", row["warnings"][0])

        # a step-1 excursion BELOW the relaxed level is not a downhill relaxation either
        row = self.diagnose(self.run_dir(energies=energies_with({1: -77.0})))
        self.assertEqual(row["severity"], "warning")
        self.assertIs(row["startup_excursion_downhill"], False)
        self.assertFalse(row["benign_warnings_only"])

    # ------------------------------------------------------------------ #
    # hard signals

    def test_non_numeric_temperature_late_in_the_run_is_hard(self) -> None:
        run = self.run_dir(temperatures=temperatures_with({350: None}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["severity"], "unstable")
        self.assertEqual(row["first_bad_step"], 350)
        self.assertEqual(row["first_bad_reasons"], ["non-numeric temperature"])
        self.assertTrue(any("non-numeric temperature at step 350" in text for text in row["hard_reasons"]))

    def test_non_numeric_free_energy_is_hard(self) -> None:
        run = self.run_dir(energies=energies_with({5: None}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 5)
        self.assertEqual(row["first_bad_reasons"], ["non-numeric free energy"])

    def test_catastrophic_energy_inside_grace_window_is_hard(self) -> None:
        run = self.run_dir(energies=energies_with({2: 2.0e6}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 2)
        self.assertEqual(len(row["first_bad_reasons"]), 1)
        self.assertIn("(catastrophic)", row["first_bad_reasons"][0])
        self.assertIn("2.0e+06 eV > 500 eV", row["first_bad_reasons"][0])
        self.assertTrue(any("catastrophic" in text for text in row["hard_reasons"]))
        # a hard row is not reported as a (benign) startup excursion
        self.assertEqual(row["startup_excursion_steps"], [])
        self.assertFalse(row["benign_warnings_only"])

    def test_multi_million_ev_runaway_is_hard_and_catastrophic(self) -> None:
        run = self.run_dir(energies=lambda step: 2.0e6 if step >= 300 else baseline_energy(step))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 300)
        self.assertIn("(catastrophic)", row["first_bad_reasons"][0])
        self.assertAlmostEqual(row["max_local_jump_ev"], 2.0e6, delta=20.0)
        self.assertEqual(row["max_local_jump_step"], 300)

    def test_sustained_post_grace_departure_is_hard(self) -> None:
        run = self.run_dir(energies=energies_with({200: 110.0, 201: 115.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 200)
        self.assertEqual(len(row["first_bad_reasons"]), 1)
        self.assertIn("eV > 50 eV (sustained)", row["first_bad_reasons"][0])
        self.assertTrue(any(text.startswith("sustained post-grace energy departure") for text in row["hard_reasons"]))
        self.assertEqual(row["isolated_spike_steps"], [])

    def test_departure_on_final_recorded_step_is_hard(self) -> None:
        run = self.run_dir(energies=energies_with({400: 110.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 400)
        self.assertIn("last recorded step", row["first_bad_reasons"][0])

    def test_startup_excursion_running_past_the_grace_window_is_hard(self) -> None:
        # Steps 9-10 lie in the grace window, but the departure carries on into steps
        # 11-12: it is one sustained departure, so the rewind anchor is its start
        # (step 9) and its grace rows are not a startup excursion.
        run = self.run_dir(energies=energies_with({9: 80.0, 10: 80.0, 11: 80.0, 12: 80.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 9)
        self.assertIn("(sustained past the grace window)", row["first_bad_reasons"][0])
        self.assertEqual(row["startup_excursion_steps"], [])
        self.assertEqual(row["warning_classes"], [])
        self.assertIsNone(row["startup_excursion_settled"])
        self.assertIn("at steps 9, 10, 11, 12", row["hard_reasons"][0])

        # a separate step-1 excursion before the departure stays a startup excursion
        run = self.run_dir(energies=energies_with({1: 77.0, 9: 80.0, 10: 80.0, 11: 80.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 9)
        self.assertEqual(row["startup_excursion_steps"], [1])

    def test_persistent_scf_ceiling_is_hard_without_rewind_anchor(self) -> None:
        run = self.run_dir(scf_iterations=lambda step: 60 if step % 2 == 0 else 3)
        row = self.diagnose(run)
        self.assertEqual(row["scf_window_steps"], 390)
        self.assertEqual(row["scf_ceiling_steps"], 195)
        self.assertGreaterEqual(row["scf_ceiling_fraction"], SCF_HARD_FRACTION)
        self.assertTrue(row["scf_unreliable"])
        self.assertTrue(row["unstable"])
        self.assertIsNone(row["first_bad_step"])
        self.assertEqual(row["first_bad_reasons"], [])
        self.assertTrue(any(text.startswith("persistent SCF failure") for text in row["hard_reasons"]))
        self.assertNotIn("scf_elevated", row["warning_classes"])

    def test_temperature_above_limit_is_hard(self) -> None:
        run = self.run_dir(temperatures=temperatures_with({100: 5000.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 100)
        self.assertEqual(row["first_bad_reasons"], ["temperature 5000 K > 1200 K"])

    def test_custom_temperature_limit_still_honoured(self) -> None:
        run = self.run_dir(temperatures=temperatures_with({100: 900.0}))
        self.assertFalse(self.diagnose(run)["unstable"])  # 900 K is only a warning at 300 K
        row = self.diagnose(run, max_temperature_k=800.0)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["temperature_limit_k"], 800.0)
        self.assertEqual(row["temperature_warning_k"], 600.0)
        self.assertEqual(row["first_bad_reasons"], ["temperature 900 K > 800 K"])

    def test_runaway_starting_inside_the_reference_window_is_anchored_at_its_onset(self) -> None:
        # Normal to step 11, 2e6 eV from step 12: the post-grace window median would be
        # the exploded level and name the healthy steps 1-11 as catastrophic.
        run = self.run_dir(energies=lambda step: 2.0e6 if step >= 12 else baseline_energy(step))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 12)
        self.assertIn("(catastrophic)", row["first_bad_reasons"][0])
        self.assertEqual(row["reference_source"], "post_grace_leading_rows")
        self.assertEqual(row["reference_window"], [11, 11])
        self.assertAlmostEqual(row["reference_free_energy_ev"], REFERENCE_F, delta=0.3)
        self.assertEqual(row["startup_excursion_steps"], [])
        self.assertEqual(len(row["hard_reasons"]), 1)
        self.assertIn("at steps 12, 13, 14", row["hard_reasons"][0])

        # a sustained +80 eV jump at step 15 (inside the window) is anchored at step 15
        run = self.run_dir(energies=energies_with({step: 80.0 for step in range(15, 401)}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 15)
        self.assertIn("(sustained)", row["first_bad_reasons"][0])
        self.assertEqual(row["reference_window"], [11, 14])
        self.assertEqual(row["warning_classes"], [])

        # a single spike inside the window does not move the reference
        row = self.diagnose(self.run_dir(energies=energies_with({15: 80.0})))
        self.assertEqual(row["reference_source"], "post_grace_window")
        self.assertEqual(row["reference_window"], [11, 20])
        self.assertEqual(row["severity"], "warning")
        self.assertEqual(row["isolated_spike_steps"], [15])

    def test_departure_next_to_a_non_numeric_energy_is_hard(self) -> None:
        run = self.run_dir(energies=energies_with({150: 80.0, 151: None}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 150)
        self.assertIn("(not an isolated spike)", row["first_bad_reasons"][0])
        self.assertEqual(row["isolated_spike_steps"], [])

    def test_torn_final_oszicar_line_is_dropped_not_read_as_a_runaway(self) -> None:
        # A healthy NiO-magnitude run killed by wall time while VASP was writing the MD
        # line of step 189: the file ends in a partial line with no newline.
        nio_level = -880.53

        def nio(step: int) -> float:
            return nio_level + 0.05 * ((step * 7) % 5)

        run = self.run_dir(nsw=400, steps=188, outcar="running", energies=nio)
        healthy = (run / "OSZICAR").read_text(encoding="utf-8")
        electronic = "RMM:   1   -0.880000E+03   -0.1E-04   -0.1E-04    10   0.1E-03\n"
        full = md_line(189, nio(189))
        torn_variants = {
            "cut inside the F mantissa": full[: full.index("F= ") + 8],
            "cut right after F=": full[: full.index("F=") + 2],
            "cut inside E": full[: full.index("E= ") + 7],
            "cut before E0=": full[: full.index(" E0=")],
        }
        for label, torn in torn_variants.items():
            with self.subTest(label):
                (run / "OSZICAR").write_text(healthy + electronic + torn, encoding="utf-8")
                row = self.diagnose(run)
                self.assertEqual(row["severity"], "ok")
                self.assertEqual(row["md_steps"], 188)
                self.assertEqual(row["last_step"], 188)
                self.assertTrue(row["torn_final_line"])
                self.assertTrue(parse_step1_oszicar(run / "OSZICAR")["torn_final_line"])

        # A final line that carries its F and E0 fields is kept even without a newline.
        for label, text in {"complete": full, "cut inside E0": full[: full.index("E0= ") + 7]}.items():
            with self.subTest(label):
                (run / "OSZICAR").write_text(healthy + electronic + text, encoding="utf-8")
                row = self.diagnose(run)
                self.assertEqual(row["md_steps"], 189)
                self.assertFalse(row["torn_final_line"])
                self.assertEqual(row["severity"], "ok")

        # ...so a genuine departure on an unterminated final line is still hard.
        run = self.run_dir(energies=energies_with({400: 110.0}))
        text = (run / "OSZICAR").read_text(encoding="utf-8")
        (run / "OSZICAR").write_text(text.rstrip("\n"), encoding="utf-8")
        row = self.diagnose(run)
        self.assertFalse(row["torn_final_line"])
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 400)

    # ------------------------------------------------------------------ #
    # corroboration

    def test_isolated_spike_with_elevated_temperature_is_corroborated_hard(self) -> None:
        run = self.run_dir(energies=energies_with({150: 80.0}), temperatures=temperatures_with({200: 700.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["corroborated"], ["isolated_energy_spike+temperature_elevated"])
        self.assertIn("corroborated anomalies: isolated_energy_spike + temperature_elevated", row["hard_reasons"])
        self.assertEqual(row["first_bad_step"], 150)
        self.assertIn("corroborated anomalies: isolated_energy_spike + temperature_elevated", row["first_bad_reasons"])
        self.assertTrue(any("isolated spike" in text for text in row["first_bad_reasons"]))

    def test_non_downhill_startup_excursion_with_elevated_temperature_is_corroborated_hard(self) -> None:
        # Step 1 BELOW F_ref: the energy rose after startup and stayed up -- not the benign relaxation.
        run = self.run_dir(energies=energies_with({1: -77.0}), temperatures=temperatures_with({200: 700.0}))
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertFalse(row["benign_warnings_only"])
        self.assertFalse(row["startup_excursion_downhill"])
        self.assertEqual(row["corroborated"], ["startup_energy_excursion+temperature_elevated"])
        self.assertEqual(row["first_bad_step"], 1)
        self.assertEqual(row["warning_classes"], ["startup_energy_excursion", "temperature_elevated"])

    def test_downhill_relaxation_finishing_just_after_the_grace_window_is_a_benign_warning(self) -> None:
        # Settling at step 11 vs 12-15 is the same physics: never a rewind to step 0.
        for onset in (11, 12, 15):
            with self.subTest(onset=onset):
                high = {step: 80.0 for step in range(1, onset)}
                row = self.diagnose(self.run_dir(energies=energies_with(high)))
                self.assertEqual(row["severity"], "warning")
                self.assertIsNone(row["first_bad_step"])
                self.assertTrue(row["benign_warnings_only"])
                self.assertEqual(row["startup_excursion_steps"], list(range(1, onset)))
                self.assertAlmostEqual(row["reference_free_energy_ev"], REFERENCE_F, delta=0.3)
        # The mirror (energy rising at step 12 and staying up) stays hard at its onset.
        rise = {step: 80.0 for step in range(12, 401)}
        row = self.diagnose(self.run_dir(energies=energies_with(rise)))
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 12)

    def test_benign_startup_relaxation_does_not_corroborate_a_distant_anomaly(self) -> None:
        # The NiO_m110 fresh-start signature plus ONE unrelated later anomaly must not
        # become hard with first_bad_step 1 (a rewind that discards the whole run).
        startup = {1: 77.0, 2: 40.0, 3: 15.0, 4: 5.0}
        cases = {
            "spike": {"energies": energies_with({**startup, 250: 60.0})},
            "warm row": {"energies": energies_with(startup), "temperatures": temperatures_with({300: 610.0})},
        }
        for label, options in cases.items():
            with self.subTest(label):
                row = self.diagnose(self.run_dir(**options))
                self.assertEqual(row["severity"], "warning")
                self.assertFalse(row["unstable"])
                self.assertEqual(row["corroborated"], [])
                self.assertIsNone(row["first_bad_step"])
                self.assertTrue(row["startup_excursion_settled"])
                self.assertTrue(row["startup_excursion_downhill"])
                self.assertFalse(row["benign_warnings_only"])  # two warning classes: still review
                self.assertEqual(len(row["warnings"]), 2)

    def test_isolated_spike_with_elevated_scf_is_corroborated_hard(self) -> None:
        run = self.run_dir(
            energies=energies_with({150: 80.0}),
            scf_iterations=lambda step: 60 if step > 10 and step % 10 < 3 else 3,
        )
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["corroborated"], ["isolated_energy_spike+scf_elevated"])
        self.assertEqual(row["first_bad_step"], 150)

    def test_elevated_temperature_with_elevated_scf_is_corroborated_hard(self) -> None:
        run = self.run_dir(
            temperatures=temperatures_with({250: 700.0}),
            scf_iterations=lambda step: 60 if step > 10 and step % 10 < 3 else 3,
        )
        row = self.diagnose(run)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["corroborated"], ["scf_elevated+temperature_elevated"])
        self.assertEqual(row["first_bad_step"], 250)
        self.assertEqual(row["first_bad_reasons"][0], "temperature 700 K > 600 K warning level")

    # ------------------------------------------------------------------ #
    # settings

    def test_teend_sets_the_temperature_target_for_a_ramp(self) -> None:
        # 100 -> 300 K ramp: judged against 300 K (warning 600 K), not TEBEG=100 K (warning 400 K).
        ramp = linear_ramp(100.0, 300.0, 400)
        run = self.run_dir(tebeg=100.0, teend=300.0, temperatures=lambda step: 500.0 if step == 250 else ramp(step))
        row = self.diagnose(run)
        self.assertEqual(row["temperature_limit_k"], 1200.0)
        self.assertEqual(row["temperature_warning_k"], 600.0)
        self.assertEqual(row["severity"], "ok")

        # 100 -> 500 K ramp: limit max(1200, 4*500) = 2000 K, so 1500 K is only a warning.
        hot_ramp = linear_ramp(100.0, 500.0, 400)
        run = self.run_dir(
            tebeg=100.0, teend=500.0, temperatures=lambda step: 1500.0 if step == 250 else hot_ramp(step)
        )
        row = self.diagnose(run)
        self.assertEqual(row["temperature_limit_k"], 2000.0)
        self.assertEqual(row["temperature_warning_k"], 1000.0)
        self.assertFalse(row["unstable"])
        self.assertEqual(row["warning_classes"], ["temperature_elevated"])

    def test_zero_grace_judges_step1_like_any_post_grace_row(self) -> None:
        energies = energies_with({1: 77.0})
        self.assertFalse(self.diagnose(self.run_dir(energies=energies))["unstable"])

        # A sustained step 1-2 excursion is hard once the grace window is disabled.
        row = self.diagnose(self.run_dir(energies=energies_with({1: 77.0, 2: 70.0})), startup_grace_steps=0)
        self.assertTrue(row["unstable"])
        self.assertEqual(row["startup_grace_steps"], 0)
        self.assertEqual(row["reference_window"], [1, 10])
        self.assertEqual(row["first_bad_step"], 1)
        self.assertIn("(sustained)", row["first_bad_reasons"][0])
        self.assertEqual(row["startup_excursion_steps"], [])
        self.assertEqual(row["scf_window_steps"], 400)

        # A lone step-1 departure has no predecessor to contradict isolation: like a
        # mid-run single-row spike it is a review-level warning, not automatic repair.
        row = self.diagnose(self.run_dir(energies=energies), startup_grace_steps=0)
        self.assertFalse(row["unstable"])
        self.assertEqual(row["severity"], "warning")
        self.assertEqual(row["warning_classes"], ["isolated_energy_spike"])
        self.assertEqual(row["isolated_spike_steps"], [1])
        self.assertFalse(row["benign_warnings_only"])
        self.assertIsNone(row["first_bad_step"])

    def test_custom_reference_window_and_energy_thresholds(self) -> None:
        run = self.run_dir(energies=energies_with({150: 30.0, 151: 30.0}))
        self.assertEqual(self.diagnose(run)["severity"], "ok")  # 30 eV is inside the 50 eV band
        row = self.diagnose(run, energy_jump_ev=20.0, reference_window_steps=5)
        self.assertEqual(row["reference_window"], [11, 15])
        self.assertTrue(row["unstable"])
        self.assertEqual(row["first_bad_step"], 150)
        row = self.diagnose(run, energy_jump_ev=20.0, catastrophic_energy_ev=25.0)
        self.assertEqual(row["catastrophic_energy_limit_ev"], 25.0)
        self.assertIn("(catastrophic)", row["first_bad_reasons"][0])

    def test_invalid_diagnostic_settings_are_rejected(self) -> None:
        run = self.run_dir(steps=12)
        with self.assertRaises(ValueError):
            diagnose_step1_run(run, startup_grace_steps=-1)
        with self.assertRaises(ValueError):
            diagnose_step1_run(run, reference_window_steps=0)
        # thresholds that would make every run look unstable (and so auto-repairable)
        for options in (
            {"energy_jump_ev": 0.0},
            {"energy_jump_ev": -5.0},
            {"energy_jump_ev": float("nan")},
            {"catastrophic_energy_ev": 25.0},  # below the 50 eV band
            {"catastrophic_energy_ev": float("nan")},
            {"max_temperature_k": -100.0},
            {"max_temperature_k": float("nan")},
        ):
            with self.subTest(options), self.assertRaises(ValueError):
                diagnose_step1_run(run, **options)

    def test_zero_temperature_limit_selects_the_default(self) -> None:
        run = self.run_dir()
        row = self.diagnose(run, max_temperature_k=0)
        self.assertEqual(row["temperature_limit_k"], 1200.0)
        self.assertEqual(row["severity"], "ok")

    def test_none_diagnostic_options_select_the_defaults(self) -> None:
        # CLI / diagnostic_options dicts may forward None for an option left unset.
        run = self.run_dir(energies=energies_with({1: 77.0}), temperatures=temperatures_with({200: 700.0}))
        expected = self.diagnose(run)
        row = self.diagnose(
            run,
            energy_jump_ev=None,
            max_temperature_k=None,
            startup_grace_steps=None,
            catastrophic_energy_ev=None,
            reference_window_steps=None,
        )
        self.assertEqual(row, expected)

    def test_constants_match_the_documented_defaults(self) -> None:
        self.assertEqual(DEFAULT_STARTUP_GRACE_STEPS, 10)
        self.assertEqual(DEFAULT_REFERENCE_WINDOW_STEPS, 10)
        self.assertEqual(DEFAULT_ENERGY_JUMP_EV, 50.0)
        self.assertEqual(DEFAULT_CATASTROPHIC_ENERGY_EV, 500.0)
        self.assertEqual(SCF_HARD_FRACTION, 0.5)
        self.assertEqual(SCF_WARN_FRACTION, 0.2)

    def test_parse_step1_oszicar_keeps_its_historical_scf_window(self) -> None:
        run = self.run_dir(steps=20, scf_iterations=lambda step: 60 if step <= 10 else 3)
        parsed = parse_step1_oszicar(run / "OSZICAR", nelm=60)
        self.assertEqual(len(parsed["steps"]), 20)
        self.assertEqual(parsed["scf_window_steps"], 15)  # steps[5:] as before
        self.assertEqual(parsed["scf_ceiling_steps"], 5)
        post_grace = parse_step1_oszicar(run / "OSZICAR", nelm=60, scf_skip_steps=10)
        self.assertEqual(post_grace["scf_window_steps"], 10)
        self.assertEqual(post_grace["scf_ceiling_steps"], 0)


class ApplyPreconditionTests(unittest.TestCase):
    def test_writes_precondition_incar_wraps_launcher_and_removes_wavecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, wavecar=True, istart=1)
            apply_precondition(run)
            pre = parse_incar(run / "INCAR.precondition")
            self.assertEqual(pre["NSW"], "0")
            self.assertEqual(pre["ISTART"], "0")
            self.assertEqual(pre["SYSTEM"], "Step1_preheat_fixture_precondition")
            launcher = (run / "runvasp.sh").read_text(encoding="utf-8")
            self.assertIn("InterfaceForge --precondition", launcher)
            self.assertEqual(launcher.count("srun -n4 vasp_std"), 2)
            self.assertFalse((run / "WAVECAR").exists())
            if os.name == "posix":
                self.assertTrue((run / "runvasp.sh").stat().st_mode & 0o111)

            # idempotent: a second call neither double-wraps nor fails
            apply_precondition(run)
            self.assertEqual((run / "runvasp.sh").read_text(encoding="utf-8"), launcher)

    def test_uses_run_slurm_when_there_is_no_runvasp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, launcher=None)
            (run / "run.slurm").write_text(PLAIN_LAUNCHER, encoding="utf-8")
            apply_precondition(run)
            self.assertIn("InterfaceForge --precondition", (run / "run.slurm").read_text(encoding="utf-8"))
            self.assertTrue((run / "INCAR.precondition").is_file())

    def test_refuses_without_a_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, launcher=None, wavecar=True)
            with self.assertRaises(SafetyError):
                apply_precondition(run)
            self.assertFalse((run / "INCAR.precondition").exists())
            self.assertTrue((run / "WAVECAR").exists())

    def test_unwrappable_launcher_is_refused_before_anything_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, wavecar=True)
            (run / "runvasp.sh").write_text("#!/bin/bash\nsrun vasp_std\nsrun vasp_gam\n", encoding="utf-8")
            before = tree_snapshot(run)
            reason = precondition_blocker(run)
            self.assertIsNotNone(reason)
            self.assertIn("exactly one line that runs vasp", reason)
            self.assertEqual(tree_snapshot(run), before)
            with self.assertRaisesRegex(SafetyError, "exactly one line that runs vasp"):
                apply_precondition(run)
            # no stray INCAR.precondition, WAVECAR kept, launcher untouched
            self.assertEqual(tree_snapshot(run), before)

    def test_precondition_blocker_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = write_step1_run(root, "plain", steps=0)
            wrapped = write_step1_run(root, "wrapped", steps=0, launcher="precondition")
            bare = write_step1_run(root, "bare", steps=0, launcher=None)
            before = tree_snapshot(root)
            self.assertIsNone(precondition_blocker(plain))
            self.assertIsNone(precondition_blocker(wrapped))  # already wrapped: nothing to refuse
            self.assertIn("needs runvasp.sh or run.slurm", precondition_blocker(bare) or "")
            self.assertEqual(tree_snapshot(root), before)

    def test_already_wrapped_launcher_is_kept_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, launcher="precondition", wavecar=True)
            launcher = (run / "runvasp.sh").read_bytes()
            self.assertEqual(
                wrap_launcher_with_precondition(launcher.decode("utf-8"), launcher_name="runvasp.sh"),
                launcher.decode("utf-8"),
            )
            apply_precondition(run)
            self.assertEqual((run / "runvasp.sh").read_bytes(), launcher)
            self.assertFalse((run / "WAVECAR").exists())

    def test_leaves_the_md_incar_to_the_caller(self) -> None:
        # ISTART=1 for the MD is the caller's job (repair/resume update_incar).
        with tempfile.TemporaryDirectory() as tmp:
            run = write_step1_run(Path(tmp), steps=0, istart=0)
            incar = (run / "INCAR").read_bytes()
            apply_precondition(run)
            self.assertEqual((run / "INCAR").read_bytes(), incar)
            self.assertEqual(parse_incar(run / "INCAR.precondition")["NSW"], "0")


if __name__ == "__main__":
    unittest.main()
