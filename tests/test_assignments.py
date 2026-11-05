from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from interfaceforge.assignments import (
    git_snapshot,
    load_assignment,
    render_assignment,
    template_placeholders,
    template_text,
    validate_report,
)
from interfaceforge.cli import main
from interfaceforge.errors import ConfigurationError
from interfaceforge.schema_check import schema_errors

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(staging: Path) -> dict:
    return {
        "id": "DD-TEST",
        "template": "due_diligence@1",
        "title": "Synthetic due diligence",
        "target": {"repo": "C:/repo", "commit": "b877871", "also_read_commits": ["8c1520f"]},
        "permissions": {
            "read": ["C:/repo"],
            "write": [f"{staging.as_posix()}/*"],
            "git_allowed": ["log", "show"],
            "forbid": ["C:/repo", "D:/forbidden"],
        },
        "tasks": [
            {"id": "T1", "kind": "judgement", "description": "Judge.", "verdicts": ["YES", "NO"], "outputs": []},
            {"id": "T2", "kind": "audit", "description": "Audit.", "command": "iface audit excess-temperature",
             "verdicts": ["CONSISTENT", "INCONSISTENT", "UNDETERMINED"], "outputs": ["audit.json"],
             "hypothesis": "Frames differ in temperature."},
            {"id": "T3", "kind": "diag_prepare", "description": "Prepare.", "command": "iface vasp ... --dry-run",
             "criterion": {"metric": "force RMSE", "unit": "meV/A", "threshold": 70, "rule": ">= 70 is material"},
             "cost": "not_inferable", "verdicts": ["PROPOSAL_WRITTEN", "BLOCKED"], "outputs": ["proposal.md"]},
        ],
        "done_when": ["Done."],
        "report_back": ["Verdicts."],
    }


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


SNAP = {"head": "b877871", "porcelain_sha256": "a" * 64, "branch_header": "## main...origin/main"}


class SchemaTests(unittest.TestCase):
    def test_package_schemas_match_repository_schemas(self) -> None:
        for name in ("assignment.schema.json", "assignment_report.schema.json",
                     "excess_temperature_input.schema.json"):
            canonical = (REPO_ROOT / "schemas" / name).read_bytes().replace(b"\r\n", b"\n")
            packaged = (REPO_ROOT / "src" / "interfaceforge" / "schemas" / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(canonical, packaged, name)

    def test_kind_specific_requirements(self) -> None:
        spec = _spec(Path("/s"))
        self.assertEqual(schema_errors(spec, "assignment.schema.json"), [])
        del spec["tasks"][2]["criterion"]
        spec["tasks"][1].pop("command")
        spec["tasks"][0]["cost"] = "from_records"
        errors = schema_errors(spec, "assignment.schema.json")
        self.assertTrue(any("/tasks/2" in e and "criterion" in e for e in errors), errors)
        self.assertTrue(any("/tasks/1" in e and "command" in e for e in errors), errors)
        self.assertTrue(any("/tasks/0" in e for e in errors), errors)

    def test_template_validates_once_placeholders_are_filled(self) -> None:
        names = template_placeholders("due_diligence_v1")
        self.assertIn("ID", names)
        values = {name: "x" for name in names}
        values.update(ID="DD-1", TARGET_COMMIT="0000000", DIAG_THRESHOLD="1")
        from string import Template

        spec = yaml.safe_load(Template(template_text("due_diligence_v1")).substitute(values))
        self.assertEqual(schema_errors(spec, "assignment.schema.json"), [])
        self.assertEqual({task["kind"] for task in spec["tasks"]}, {"audit", "diag_prepare", "judgement"})


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "DD-TEST"
        self.root.mkdir()
        (self.root / "assignment.yaml").write_text(yaml.safe_dump(_spec(self.root), sort_keys=False),
                                                   encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _render_and_freeze(self) -> dict:
        self.assertEqual(_run_cli(["assignment", "render", str(self.root / "assignment.yaml")])[0], 0)
        code, out, err = _run_cli(["assignment", "freeze", str(self.root)])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def _report(self, **overrides) -> Path:
        frozen = json.loads((self.root / "frozen.json").read_text(encoding="utf-8"))
        written = self.root / "audit.json"
        written.write_text("{}", encoding="utf-8")
        report = {
            "assignment_id": "DD-TEST",
            "frozen_sha256": frozen["assignment_yaml_sha256"],
            "agent": {"name": "test", "date": "2026-11-05"},
            "git_state": {"before": dict(SNAP), "after": dict(SNAP)},
            "files_written": [{"path": str(written),
                               "sha256": hashlib.sha256(written.read_bytes()).hexdigest()}],
            "tasks": [
                {"id": "T1", "verdict": "YES", "findings": []},
                {"id": "T2", "verdict": "UNDETERMINED", "findings": [
                    {"label": "E", "text": "ratio", "values": [{"value": 1.019, "unit": "", "source": "a.json"}]}
                ]},
                {"id": "T3", "verdict": "PROPOSAL_WRITTEN", "findings": []},
            ],
            "not_assessed": [],
            "deviations": ["none"],
        }
        report.update(overrides)
        path = self.root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_render_is_deterministic_and_has_handoff_sections(self) -> None:
        out_a, out_b = self.root / "a.md", self.root / "b.md"
        for target in (out_a, out_b):
            self.assertEqual(
                _run_cli(["assignment", "render", str(self.root / "assignment.yaml"), "-o", str(target)])[0], 0
            )
        self.assertEqual(hashlib.sha256(out_a.read_bytes()).hexdigest(),
                         hashlib.sha256(out_b.read_bytes()).hexdigest())
        text = out_a.read_text(encoding="utf-8")
        for marker in ("=====", "TASK: ", "HARD RULES", "TASK 1 [T1]", "DELIVERABLES", "DONE WHEN", "REPORT BACK",
                       "Pre-registered criterion: force RMSE [meV/A], threshold 70", "Git: only log, show"):
            self.assertIn(marker, text)
        self.assertEqual(text, render_assignment(load_assignment(self.root / "assignment.yaml")))

    def test_freeze_records_hashes_ledger_and_refuses_second_freeze(self) -> None:
        frozen = self._render_and_freeze()
        self.assertEqual(frozen["assignment_yaml_sha256"],
                         hashlib.sha256((self.root / "assignment.yaml").read_bytes()).hexdigest())
        self.assertTrue((self.root / "NOTES.md").is_file())
        state = json.loads((self.root / ".interfaceforge" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["events"][-1]["action"], "frozen")
        before = (self.root / "frozen.json").read_bytes()
        code, _, err = _run_cli(["assignment", "freeze", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual((self.root / "frozen.json").read_bytes(), before)

    def test_freeze_refuses_a_stale_render(self) -> None:
        _run_cli(["assignment", "render", str(self.root / "assignment.yaml")])
        with (self.root / "ASSIGNMENT.md").open("a", encoding="utf-8") as handle:
            handle.write("edited\n")
        code, _, err = _run_cli(["assignment", "freeze", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("does not match", err)

    def test_valid_report(self) -> None:
        self._render_and_freeze()
        code, out, _ = _run_cli(["assignment", "validate", str(self.root), "--report", str(self._report())])
        self.assertEqual(code, 0)
        verdict = json.loads(out)
        self.assertTrue(verdict["valid"], verdict["failures"])

    def test_partial_report_lists_missing_tasks_and_bad_verdicts(self) -> None:
        self._render_and_freeze()
        path = self._report(tasks=[{"id": "T1", "verdict": "MAYBE", "findings": [
            {"label": "E", "text": "x", "values": [{"value": 1, "unit": "eV", "source": " "}]}]}])
        code, out, _ = _run_cli(["assignment", "validate", str(self.root), "--report", str(path)])
        self.assertEqual(code, 0)
        verdict = json.loads(out)
        self.assertFalse(verdict["valid"])
        failures = "\n".join(verdict["failures"])
        self.assertIn("task T2: no verdict", failures)
        self.assertIn("task T3: no verdict", failures)
        self.assertIn("'MAYBE' not in", failures)
        self.assertIn("empty source", failures)
        self.assertEqual(verdict["tasks"]["T2"]["status"], "missing")

    def test_git_state_and_permission_failures(self) -> None:
        self._render_and_freeze()
        moved = dict(SNAP, head="c0ffee1", branch_header="## main...origin/main [behind 1]")
        outside = Path(self._tmp.name) / "elsewhere.txt"
        outside.write_text("x", encoding="utf-8")
        path = self._report(
            git_state={"before": dict(SNAP), "after": moved},
            files_written=[{"path": str(outside), "sha256": "0" * 64}],
        )
        failures = "\n".join(validate_report(self.root, path)["failures"])
        self.assertIn("HEAD moved", failures)
        self.assertIn("branch_header changed", failures)
        self.assertIn("matches no permissions.write glob", failures)
        self.assertIn("sha256 does not match", failures)

    def test_branch_header_change_explained_by_deviation(self) -> None:
        self._render_and_freeze()
        path = self._report(
            git_state={"before": dict(SNAP), "after": dict(SNAP, branch_header="## main [behind 2]")},
            deviations=["branch_header moved by another process's fetch"],
        )
        self.assertTrue(validate_report(self.root, path)["valid"])

    def test_edit_after_freeze_and_wrong_hash_detected(self) -> None:
        self._render_and_freeze()
        path = self._report(frozen_sha256="f" * 64)
        with (self.root / "assignment.yaml").open("a", encoding="utf-8") as handle:
            handle.write("# sneaky edit\n")
        failures = "\n".join(validate_report(self.root, path)["failures"])
        self.assertIn("frozen_sha256", failures)
        self.assertIn("edited after freezing", failures)

    def test_missing_required_report_fields_fail_schema(self) -> None:
        self._render_and_freeze()
        report = json.loads(self._report().read_text(encoding="utf-8"))
        del report["deviations"]
        del report["not_assessed"]
        path = self.root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        failures = "\n".join(validate_report(self.root, path)["failures"])
        self.assertIn("deviations", failures)
        self.assertIn("not_assessed", failures)

    def test_unreadable_report_exits_2(self) -> None:
        self._render_and_freeze()
        (self.root / "bad.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(_run_cli(["assignment", "validate", str(self.root), "--report",
                                   str(self.root / "bad.json")])[0], 2)
        self.assertEqual(_run_cli(["assignment", "validate", str(self.root), "--report",
                                   str(self.root / "missing.json")])[0], 2)

    def test_invalid_spec_is_rejected(self) -> None:
        spec = _spec(self.root)
        spec["tasks"].append(dict(spec["tasks"][0]))
        (self.root / "assignment.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            load_assignment(self.root / "assignment.yaml")

    def test_init_from_template(self) -> None:
        target = Path(self._tmp.name) / "fresh"
        code, out, _ = _run_cli(
            ["assignment", "init", str(target), "--set", "ID=DD-9", "--set", "TARGET_COMMIT=abcdef1"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("TITLE", payload["unfilled_placeholders"])
        self.assertIn('id: "DD-9"', (target / "assignment.yaml").read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitSnapshotTests(unittest.TestCase):
    def test_snapshot_hashes_porcelain_without_branch_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)  # noqa: E731
            run("init", "-q")
            run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
            clean = git_snapshot(tmp)
            self.assertEqual(clean["porcelain_sha256"], hashlib.sha256(b"").hexdigest())
            self.assertTrue(clean["branch_header"].startswith("## "))
            (Path(tmp) / "new.txt").write_text("x", encoding="utf-8")
            dirty = git_snapshot(tmp)
            self.assertEqual(dirty["head"], clean["head"])
            self.assertEqual(dirty["porcelain_sha256"], hashlib.sha256(b"?? new.txt\n").hexdigest())


if __name__ == "__main__":
    unittest.main()
