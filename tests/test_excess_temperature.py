from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.cli import main
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.excess_temperature import (
    INCONSISTENT,
    UNDETERMINED,
    audit_case,
    equipartition_check,
    run_audit,
)
from interfaceforge.interface_mu import K_B_EV_PER_K
from interfaceforge.validation import EV_A2_TO_J_M2

TIN_THERMAL = -20.245090
TIN_STATIC = -20.324136
D_IDEAL = 803.297


def _case(frame_temperatures: list[float | None], *, bulk_t: tuple[float, float] = (300.0, 300.0)) -> dict:
    return {
        "label": "synthetic",
        "interface_frames": [
            {"id": f"f{index}", "family": "ideal", "n_atoms": 300, "T_K": t_k, "coeff": 1.0}
            for index, t_k in enumerate(frame_temperatures)
        ],
        "bulk_refs": [
            {"name": "TiN", "e_per_fu_eV": TIN_THERMAL, "atoms_per_fu": 2, "T_K": bulk_t[0],
             "kind": "thermal_mean", "n_fu_coeff": 140},
            {"name": "Si3N4", "e_per_fu_eV": -58.0045, "atoms_per_fu": 7, "T_K": bulk_t[1],
             "kind": "thermal_mean", "n_fu_coeff": 48},
        ],
        "static_refs": [
            {"name": "TiN", "e_per_fu_eV": TIN_STATIC},
            {"name": "Si3N4", "e_per_fu_eV": -58.280429},
        ],
        "denominator_A2": D_IDEAL,
    }


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class EquipartitionTests(unittest.TestCase):
    def test_tin_known_answer(self) -> None:
        result = equipartition_check(TIN_THERMAL, TIN_STATIC, 300.0, 2, tol_ratio=0.10)
        self.assertAlmostEqual(result["delta_eV_fu"] * 1e3, 79.046, places=6)
        self.assertAlmostEqual(result["expected_eV_fu"] * 1e3, 77.556, places=3)
        self.assertAlmostEqual(K_B_EV_PER_K, 8.617333262e-5, places=15)
        self.assertAlmostEqual(result["ratio"], 1.019, places=3)
        self.assertEqual(result["status"], "PASS")

    def test_ratio_outside_tolerance_fails(self) -> None:
        # A static energy passed as the "thermal mean" gives ratio 0.
        self.assertEqual(equipartition_check(TIN_STATIC, TIN_STATIC, 300.0, 2, tol_ratio=0.10)["status"], "FAIL")


class TemperatureMatchTests(unittest.TestCase):
    def test_synthetic_mismatch_reproduces_analytic_bias(self) -> None:
        result = audit_case(_case([300.0, 450.0]))
        analytic = 300 * 1.5 * K_B_EV_PER_K * 150 / 803.297 * EV_A2_TO_J_M2
        self.assertAlmostEqual(result["delta_gamma_J_m2"], analytic, delta=1e-9)
        self.assertAlmostEqual(result["delta_gamma_J_m2"], 0.116, places=3)
        self.assertEqual(result["verdict"], INCONSISTENT)
        self.assertEqual(result["c1_status"], "PASS")

    def test_all_frames_at_reference_temperature_is_consistent(self) -> None:
        result = audit_case(_case([300.0, 300.0]))
        self.assertEqual(result["verdict"], "CONSISTENT")
        self.assertEqual(result["delta_gamma_J_m2"], 0.0)

    def test_null_frame_temperature_is_undetermined_with_reason(self) -> None:
        result = audit_case(_case([300.0, None]))
        self.assertEqual(result["verdict"], UNDETERMINED)
        self.assertIsNone(result["delta_gamma_J_m2"])
        self.assertTrue(any("T_K is null" in reason and "f1" in reason for reason in result["reasons"]))

    def test_missing_static_reference_is_undetermined(self) -> None:
        case = _case([300.0, 300.0])
        case["static_refs"] = case["static_refs"][:1]
        result = audit_case(case)
        self.assertEqual(result["verdict"], UNDETERMINED)
        self.assertEqual(result["c1_status"], "INCOMPLETE")

    def test_c3_spread_per_family(self) -> None:
        case = _case([300.0, 300.0])
        case["gamma_values"] = [
            {"family": "ideal", "gamma_J_m2": 5.0},
            {"family": "ideal", "gamma_J_m2": 5.2},
            {"family": "real", "gamma_J_m2": 4.3},
        ]
        spread = audit_case(case)["c3_spread"]
        self.assertAlmostEqual(spread["ideal"]["mean_J_m2"], 5.1)
        self.assertAlmostEqual(spread["ideal"]["std_J_m2"], 0.1414213562, places=9)
        self.assertAlmostEqual(spread["ideal"]["sem_J_m2"], 0.1, places=9)
        self.assertIsNone(spread["real"]["std_J_m2"])

    def test_bulk_refs_at_different_temperatures_is_an_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            audit_case(_case([300.0, 300.0], bulk_t=(300.0, 450.0)))


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, payload: dict, name: str = "input.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_cli_writes_stage_outputs_and_ledger_event(self) -> None:
        path = self._write(_case([300.0, 450.0]))
        code, out, _ = _run_cli(["audit", "excess-temperature", "--input", str(path)])
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual(summary["verdict"], INCONSISTENT)
        for name in ("excess_temperature_manifest.json", "excess_temperature_audit.json",
                     "excess_temperature_audit.tsv", "excess_temperature_audit.md"):
            self.assertTrue((self.root / name).is_file(), name)
        manifest = json.loads((self.root / "excess_temperature_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["input_sha256"], summary["input_sha256"])
        state = json.loads((self.root / ".interfaceforge" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["events"][-1]["action"], "excess_temperature_audit")
        tsv = (self.root / "excess_temperature_audit.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(tsv), 3)

    def test_overwrite_refused_then_forced(self) -> None:
        path = self._write(_case([300.0, 300.0]))
        self.assertEqual(_run_cli(["audit", "excess-temperature", "--input", str(path)])[0], 0)
        code, _, err = _run_cli(["audit", "excess-temperature", "--input", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("ERROR:", err)
        self.assertIn("--force", err)
        with self.assertRaises(SafetyError):
            run_audit(path)
        code, out, _ = _run_cli(["audit", "excess-temperature", "--input", str(path), "--force"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["verdict"], "CONSISTENT")

    def test_bulk_temperature_disagreement_exits_2(self) -> None:
        path = self._write(_case([300.0], bulk_t=(300.0, 450.0)))
        code, out, err = _run_cli(["audit", "excess-temperature", "--input", str(path)])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("disagree on temperature", err)

    def test_schema_violation_exits_2(self) -> None:
        bad = _case([300.0])
        del bad["denominator_A2"]
        code, _, err = _run_cli(["audit", "excess-temperature", "--input", str(self._write(bad))])
        self.assertEqual(code, 2)
        self.assertIn("excess_temperature_input.schema.json", err)

    def test_multiple_cases_take_the_worst_verdict(self) -> None:
        path = self._write({"schema_version": 1, "cases": [_case([300.0]), _case([None])]})
        code, out, _ = _run_cli(["audit", "excess-temperature", "--input", str(path), "-o", str(self.root / "o")])
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual([case["verdict"] for case in summary["cases"]], ["CONSISTENT", UNDETERMINED])
        self.assertEqual(summary["verdict"], UNDETERMINED)

    def test_plain_audit_command_still_takes_a_root(self) -> None:
        from interfaceforge.cli import build_parser, parse_cli_args

        args = parse_cli_args(build_parser(), ["audit", str(self.root)])
        self.assertEqual(args.command, "audit")
        self.assertEqual(args.root, str(self.root))


if __name__ == "__main__":
    unittest.main()
