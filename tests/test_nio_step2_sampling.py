"""step2_sample.json consistency: explicit, auditable sampling provenance for the canonical dataset."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ase.io import read
from nio_fixture import build_nio_tree, outcar_text

from interfaceforge.cli import main as cli_main
from interfaceforge.errors import SafetyError
from interfaceforge.nio_dataset import ExportConfig, export_dataset, plan_export, verify_dataset
from interfaceforge.nio_readiness import backend_consumption, readiness_audit
from interfaceforge.state import sha256_file

CASE = "NiO_m110_Big_U46_OH50_clustered_dissoc"
CASES = [
    ("OH0", "NiO_m110_Big_U46", ("Ni", "O"), (4, 4)),
    ("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6)),
    ("OH50", "NiO_m110_Big_U46_OH50_clustered_capped", ("H", "Ni", "O"), (2, 4, 6)),
]
TID = f"NiO_head/Step2_300K/OH50/{CASE}"


class _SamplingCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def tree(self, special: dict | None = None) -> Path:
        self.head = build_nio_tree(self.root / "NiO_head", cases=CASES, temperatures=(300,), special=special or {})
        self.manifest = self.head / "Step2_300K" / "step2_sample.json"
        return self.head

    def edit_manifest(self, mutate) -> None:
        data = json.loads(self.manifest.read_text())
        mutate(data)
        self.manifest.write_text(json.dumps(data))

    def set_case_row(self, **fields) -> None:
        def mutate(data):
            row = next(item for item in data["runs"] if item["relative_path"].endswith(CASE))
            row.update(fields)

        self.edit_manifest(mutate)

    def analysis(self, config: ExportConfig | None = None):
        plan = plan_export([self.head], config or ExportConfig(), require_usable=False)
        return plan, next(item for item in plan.analyses if item.trajectory.trajectory_id == TID)

    def assert_blocked(self, *needles: str, affected: int = 1) -> dict:
        payload = readiness_audit([self.head], ExportConfig(), output=self.root / "ready")
        self.assertFalse(payload["ready_for_gpu_smoke_tests"])
        blocking = "\n".join(payload["blocking_items"])
        self.assertIn("step2_sample.json", blocking)
        self.assertIn(f"for {affected} trajectory(ies)", blocking)
        if affected == 1:
            self.assertIn(TID, blocking)
        problems = payload["answers"]["step2_sampling"]["problems"]
        self.assertIn(TID, {problem["trajectory_id"] for problem in problems})
        self.assertEqual(len(problems), affected)
        for needle in needles:
            self.assertIn(needle, blocking)
        self.assertTrue(payload["answers"]["step2_sampling"]["blocking"])
        with self.assertRaises(SafetyError) as refused:
            export_dataset([self.head], self.root / "dataset", ExportConfig())
        message = str(refused.exception)
        self.assertIn("refusing to write a canonical dataset", message)
        self.assertIn(TID, message)
        self.assertIn(str(self.manifest), message)
        for needle in needles:
            self.assertIn(needle, message)
        self.assertFalse((self.root / "dataset" / "manifest.json").exists())
        return payload


class ConsistentSamplingTests(_SamplingCase):
    def test_all_requested_frames_present_and_passing(self) -> None:
        self.tree()
        _, analysis = self.analysis()
        self.assertEqual(analysis.sample_state, "used")
        self.assertEqual(analysis.selection, "step2_sample")
        self.assertEqual(analysis.sample_indices_requested, [0, 2, 4, 6])
        self.assertEqual(analysis.sample_indices_present, [0, 2, 4, 6])
        self.assertEqual(analysis.sample_indices_selected, [0, 2, 4, 6])
        self.assertEqual(analysis.sample_indices_rejected_by_qc, [])
        self.assertEqual(analysis.sample_indices_missing, [])
        self.assertEqual(analysis.sampled_indices, 4)  # legacy field retained
        self.assertEqual(analysis.sampling_problems, [])

        out = Path(export_dataset([self.head], self.root / "dataset", ExportConfig())["output_root"])
        with (out / "trajectories.csv").open() as handle:
            row = next(item for item in csv.DictReader(handle) if item["trajectory_id"] == TID)
        self.assertEqual(row["sample_state"], "used")
        self.assertEqual(row["sample_manifest"], str(self.manifest))
        self.assertEqual(row["sample_manifest_sha256"], sha256_file(self.manifest))
        self.assertEqual(row["step2_sample_indices"], "4")
        for key, value in {
            "requested": "4",
            "present": "4",
            "rejected_by_qc": "0",
            "selected": "4",
            "missing": "0",
            "exported": "4",
        }.items():
            self.assertEqual(row[f"sample_count_{key}"], value, key)
        self.assertEqual(row["sample_indices_requested"], "0 2 4 6")
        self.assertEqual(row["sample_indices_selected"], "0 2 4 6")
        manifest = json.loads((out / "manifest.json").read_text())
        totals = manifest["sampling"]["totals"]
        self.assertEqual(totals["requested"], 12)
        self.assertEqual(totals["exported"], sum(manifest["frame_counts"].values()))
        self.assertEqual(manifest["sampling"]["manifests"], {str(self.manifest): sha256_file(self.manifest)})
        ready = readiness_audit([self.head], ExportConfig(), output=self.root / "ready", dataset=out)
        self.assertTrue(ready["ready_for_gpu_smoke_tests"], ready["blocking_items"])
        self.assertEqual(ready["attention_items"], [])

    def test_valid_sampling_export_is_deterministic_and_backend_consistent(self) -> None:
        self.tree()
        first = export_dataset([self.head], self.root / "a", ExportConfig())
        second = export_dataset([self.head], self.root / "b", ExportConfig())
        self.assertEqual(first["dataset_hash"], second["dataset_hash"])
        self.assertEqual(first["split_hash"], second["split_hash"])
        report = verify_dataset(self.root / "a")
        self.assertTrue(report["valid"], report["problems"])
        self.assertTrue(all(item["exact_membership"] for item in report["extxyz_deepmd_membership"].values()))
        consumption = backend_consumption(None, self.root / "a")
        self.assertTrue(consumption["all_enabled_backends_share_dataset_and_split"])
        self.assertEqual(set(consumption["backends"]), {"mace", "deepmd", "nequip"})
        manifest = json.loads((self.root / "a" / "manifest.json").read_text())
        mace, nequip = manifest["backends"]["mace"], manifest["backends"]["nequip"]
        self.assertEqual(
            (mace["train_file"], mace["valid_file"], mace["test_file"]),
            (nequip["train_file_path"], nequip["val_file_path"], nequip["test_file_path"]),
        )


class MissingFrameTests(_SamplingCase):
    def test_requested_frame_beyond_the_trajectory_blocks(self) -> None:
        self.tree()
        self.set_case_row(indices=[0, 2, 4, 6, 900], kept_frames=5)
        _, analysis = self.analysis()
        self.assertEqual(analysis.status, "sampling_inconsistent")
        self.assertEqual(analysis.sample_indices_missing, [900])
        self.assertEqual(analysis.sample_indices_present, [0, 2, 4, 6])
        self.assertEqual(analysis.sampling_problems[0]["parsed_frames"], 8)
        payload = self.assert_blocked("900", "8 parsed frame(s), indices 0-7")
        per = next(row for row in payload["answers"]["step2_sampling"]["per_trajectory"] if row["trajectory_id"] == TID)
        self.assertEqual((per["requested"], per["present"], per["missing"], per["exported"]), (5, 4, 1, 0))
        self.assertIn("MISSING FRAMES", (self.root / "ready" / "readiness.md").read_text())

    def test_parser_stopping_before_a_requested_frame_blocks(self) -> None:
        self.tree(special={f"Step2_300K/{CASE}": {"truncate_last": True}})
        self.set_case_row(indices=[0, 2, 4, 7], kept_frames=4)
        _, analysis = self.analysis()
        self.assertEqual(analysis.frames_parsed, 7)
        self.assertEqual(analysis.sample_indices_missing, [7])
        self.assert_blocked("7 parsed frame(s), indices 0-6")

    def test_cli_export_exits_nonzero_with_the_reason(self) -> None:
        self.tree()
        self.set_case_row(indices=[0, 900], kept_frames=2)
        stderr = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            code = cli_main(["dataset", "export", str(self.head), "--output", str(self.root / "cli")])
        self.assertEqual(code, 2)
        self.assertIn("900", stderr.getvalue())
        buffer = StringIO()
        with redirect_stdout(buffer):
            cli_main(["dataset", "readiness", str(self.head), "--output", str(self.root / "r")])
        summary = json.loads(buffer.getvalue())
        self.assertFalse(summary["ready_for_gpu_smoke_tests"])
        self.assertTrue(summary["step2_sampling"]["blocking"])


class QcRejectionTests(_SamplingCase):
    def test_scf_rejected_request_is_reported_not_forced(self) -> None:
        self.tree(special={f"Step2_300K/{CASE}": {"scf_ceiling_steps": (3,)}})
        _, analysis = self.analysis()
        self.assertEqual(analysis.sample_state, "used")
        self.assertEqual(analysis.sample_indices_requested, [0, 2, 4, 6])
        self.assertEqual(analysis.sample_indices_present, [0, 2, 4, 6])
        self.assertEqual(analysis.sample_indices_rejected_by_qc, [2])
        self.assertEqual(analysis.sample_indices_selected, [0, 4, 6])
        self.assertEqual(analysis.sample_indices_missing, [])
        self.assertEqual(analysis.sampling_problems, [])
        out = Path(export_dataset([self.head], self.root / "dataset", ExportConfig())["output_root"])
        exported = [
            int(atoms.info["source_frame"])
            for split in ("train", "valid", "test")
            if (out / f"{split}.extxyz").is_file()
            for atoms in read(out / f"{split}.extxyz", index=":")
            if atoms.info["IF_leaf"] == TID
        ]
        self.assertEqual(sorted(exported), [0, 4, 6])
        payload = readiness_audit([self.head], ExportConfig(), output=self.root / "ready", dataset=out)
        self.assertTrue(payload["ready_for_gpu_smoke_tests"], payload["blocking_items"])
        self.assertTrue(any("failed QC" in item for item in payload["attention_items"]))
        rejected = payload["answers"]["step2_sampling"]["qc_rejected_requested_frames"]
        self.assertEqual(rejected[0]["trajectory_id"], TID)
        self.assertEqual(rejected[0]["index"], 2)
        self.assertIn("SCF not converged", rejected[0]["reason"])
        markdown = (self.root / "ready" / "readiness.md").read_text()
        self.assertIn(f"| `{TID}` | yes | 4 | 4 | 1 | 3 | 0 | used |", markdown)
        self.assertIn("Needs attention (not blocking)", markdown)

    def test_requests_after_a_temperature_runaway_are_rejected_by_qc(self) -> None:
        self.tree(special={f"Step2_300K/{CASE}": {"runaway_from": 5}})
        plan, analysis = self.analysis()
        self.assertEqual(analysis.sample_indices_rejected_by_qc, [4, 6])
        self.assertEqual(analysis.sample_indices_selected, [0, 2])
        self.assertEqual(analysis.sample_indices_missing, [])
        reasons = {row["source_frame"]: row["reason"] for row in analysis.rejected}
        self.assertIn("temperature runaway", reasons[4])
        sampling = [problem for item in plan.analyses for problem in item.sampling_problems]
        self.assertEqual(sampling, [])


class MalformedManifestTests(_SamplingCase):
    def _assert_invalid(self, needle: str, affected: int = 1) -> None:
        _, analysis = self.analysis()
        self.assertEqual(analysis.sample_state, "invalid")
        self.assertEqual(analysis.status, "sampling_invalid")
        self.assertEqual(analysis.selection, "step2_sample", "an invalid manifest must never fall back to stride")
        self.assertEqual(analysis.frames, [])
        self.assertIn(needle, " ".join(analysis.sample_errors))
        self.assert_blocked("invalid", affected=affected)

    def test_non_integer_index(self) -> None:
        self.tree()
        self.set_case_row(indices=[0, "2", 4.5], kept_frames=3)
        self._assert_invalid("non-integer")

    def test_negative_index(self) -> None:
        self.tree()
        self.set_case_row(indices=[-1, 2], kept_frames=2)
        self._assert_invalid("negative")

    def test_duplicate_indices_are_rejected_not_deduplicated(self) -> None:
        self.tree()
        self.set_case_row(indices=[0, 2, 2], kept_frames=3)
        self._assert_invalid("duplicate")

    def test_ok_entry_without_indices(self) -> None:
        self.tree()
        self.edit_manifest(
            lambda data: next(r for r in data["runs"] if r["relative_path"].endswith(CASE)).pop("indices")
        )
        self._assert_invalid("'indices' must be a list")

    def test_kept_frames_disagreeing_with_indices(self) -> None:
        self.tree()
        self.set_case_row(indices=[0, 2], kept_frames=4)
        self._assert_invalid("kept_frames")

    def test_ambiguous_entries_are_invalid_but_identical_duplicates_are_accepted(self) -> None:
        self.tree()

        def add_conflict(data):
            row = next(item for item in data["runs"] if item["relative_path"].endswith(CASE))
            data["runs"].append({**row, "indices": [1, 3], "kept_frames": 2})

        self.edit_manifest(add_conflict)
        self._assert_invalid("ambiguous")

        self.tree()
        self.edit_manifest(
            lambda data: data["runs"].append(dict(next(r for r in data["runs"] if r["relative_path"].endswith(CASE))))
        )
        _, analysis = self.analysis()
        self.assertEqual(analysis.sample_state, "used")
        self.assertTrue(any("identical" in warning for warning in analysis.warnings))

    def test_malformed_file_level_structure(self) -> None:
        for label, content, needle in (
            ("json", "{not json", "not valid JSON"),
            ("runs", json.dumps({"runs": {"a": 1}}), "malformed runs"),
            ("row", json.dumps({"runs": [17]}), "malformed runs[0]"),
            ("format", json.dumps({"format": "something-else", "runs": []}), "format"),
        ):
            with self.subTest(label):
                self.tree()
                self.manifest.write_text(content)
                # A broken file invalidates every run in its Step2 root, never silently.
                self._assert_invalid(needle, affected=len(CASES))


class NoOrDisabledManifestTests(_SamplingCase):
    def test_no_manifest_keeps_stride_selection(self) -> None:
        self.tree()
        self.manifest.unlink()
        _, analysis = self.analysis()
        self.assertEqual((analysis.sample_state, analysis.selection), ("absent", "stride"))
        self.assertEqual([frame.source_frame for frame in analysis.frames], list(range(8)))
        self.assertIsNone(analysis.sample_indices_requested)
        self.assertIsNone(analysis.sampled_indices)
        _, strided = self.analysis(ExportConfig(stride=3))
        self.assertEqual([frame.source_frame for frame in strided.frames], [0, 3, 6])
        payload = readiness_audit([self.head], ExportConfig(), output=self.root / "ready")
        self.assertFalse(payload["answers"]["step2_sampling"]["blocking"])
        self.assertIn("no (stride)", (self.root / "ready" / "readiness.md").read_text())

    def test_disabled_sampling_ignores_even_a_broken_manifest(self) -> None:
        self.tree()
        self.manifest.write_text("{not json")
        _, analysis = self.analysis(ExportConfig(use_step2_sample=False))
        self.assertEqual((analysis.sample_state, analysis.selection), ("disabled", "stride"))
        self.assertEqual(analysis.sampling_problems, [])

    def test_run_missing_from_or_pending_in_the_manifest_is_stale(self) -> None:
        self.tree()
        self.edit_manifest(
            lambda data: data.update(runs=[r for r in data["runs"] if not r["relative_path"].endswith(CASE)])
        )
        _, analysis = self.analysis()
        self.assertEqual((analysis.sample_state, analysis.status), ("stale", "sampling_invalid"))
        self.assertEqual(analysis.frames, [])
        self.assert_blocked("does not list this run")

        self.tree()
        self.set_case_row(status="PENDING", indices=None)
        _, analysis = self.analysis()
        self.assertEqual(analysis.sample_state, "stale")
        self.assert_blocked("'PENDING'")

    def test_pending_entry_for_a_run_without_frames_is_consistent(self) -> None:
        self.tree()
        self.set_case_row(status="PENDING", indices=None)
        (self.head / "Step2_300K" / "OH50" / CASE / "OUTCAR").write_text(outcar_text(("H", "Ni", "O"), (4, 4, 6), []))
        _, analysis = self.analysis()
        self.assertEqual((analysis.sample_state, analysis.status), ("pending", "empty"))
        self.assertEqual(analysis.sampling_problems, [])


if __name__ == "__main__":
    unittest.main()
