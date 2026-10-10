"""Characterization tests for exploration, report and vasp_provenance (synthetic inputs only)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from interfaceforge import report, vasp_provenance
from interfaceforge.config import Campaign, SystemSpec
from interfaceforge.exploration import generate_exploration
from interfaceforge.state import StateStore


def make_campaign(
    root: Path,
    *,
    systems: tuple[SystemSpec, ...] | None = None,
    exploration: dict | None = None,
    name: str = "demo",
    description: str = "a test campaign",
) -> Campaign:
    if systems is None:
        systems = (SystemSpec(id="interface", kind="interface", structure=root / "POSCAR"),)
    return Campaign(
        path=root / "campaign.yaml",
        root=root,
        name=name,
        description=description,
        profile_path=root / "profile.yaml",
        systems=systems,
        reference={},
        stages={},
        dataset={},
        models={},
        active_learning={},
        exploration=exploration or {},
        validation={},
        raw={},
    )


class ExplorationTests(unittest.TestCase):
    def test_defaults_expand_two_temperatures_with_deterministic_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = generate_exploration(make_campaign(root))
            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["temperatures_k"], [300.0, 450.0])
            self.assertEqual([task["seed"] for task in payload["tasks"]], [20260730, 20260730 + 7919])
            self.assertEqual({task["status"] for task in payload["tasks"]}, {"planned"})
            self.assertEqual(generate_exploration(make_campaign(root))["tasks"], payload["tasks"])
            output = root / "runs" / "exploration"
            with (output / "tasks.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]),
                ["task_id", "system", "kind", "temperature_k", "strain", "replica", "seed", "status"],
            )
            self.assertEqual(json.loads((output / "manifest.json").read_text())["campaign"], "demo")

    def test_product_order_is_system_temperature_strain_replica(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            systems = tuple(
                SystemSpec(id=name, kind="interface", structure=root / "P") for name in ("a", "b")
            )
            campaign = make_campaign(
                root,
                systems=systems,
                exploration={"temperatures": [300, 600], "strains": [0.0, 0.02], "replicas": 2},
            )
            tasks = generate_exploration(campaign, output=root / "out")["tasks"]
            self.assertEqual(len(tasks), 16)
            first = [(t["system"], t["temperature_k"], t["strain"], t["replica"]) for t in tasks[:5]]
            self.assertEqual(
                first,
                [("a", 300.0, 0.0, 0), ("a", 300.0, 0.0, 1), ("a", 300.0, 0.02, 0),
                 ("a", 300.0, 0.02, 1), ("a", 600.0, 0.0, 0)],
            )
            self.assertEqual(tasks[8]["system"], "b")
            self.assertEqual([t["task_id"] for t in tasks], list(range(16)))
            self.assertTrue((root / "out" / "tasks.csv").is_file())

    def test_records_state_event_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_exploration(make_campaign(root))
            state = StateStore(root).load()
            self.assertEqual(state["events"][-1]["action"], "explore")
            self.assertEqual(state["events"][-1]["details"]["tasks"], 2)
            self.assertIn("exploration_manifest", state["artifacts"])

    def test_non_positive_replicas_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "replicas must be positive"):
                generate_exploration(make_campaign(Path(temporary), exploration={"replicas": 0}))

    def test_campaign_without_systems_raises_index_error(self) -> None:
        # CHARACTERIZATION (suspected bug, low): an empty task list reaches ``tasks[0]`` when
        # building the CSV header and fails with a bare IndexError instead of a clear message.
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(IndexError):
                generate_exploration(make_campaign(Path(temporary), systems=()))


class ReportTests(unittest.TestCase):
    def test_table_escapes_html_and_blanks_none(self) -> None:
        text = report._table(["<h>"], [["<b>&", None], [0]])
        self.assertIn("<th>&lt;h&gt;</th>", text)
        self.assertIn("<td>&lt;b&gt;&amp;</td><td></td>", text)
        self.assertIn("<td>0</td>", text)

    def test_load_json_tolerates_missing_and_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad.json").write_text("{not json", encoding="utf-8")
            (root / "good.json").write_text('{"a": 1}', encoding="utf-8")
            self.assertIsNone(report._load_json(root / "missing.json"))
            self.assertIsNone(report._load_json(root / "bad.json"))
            self.assertEqual(report._load_json(root / "good.json"), {"a": 1})

    def test_lcurve_final_reads_the_last_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lcurve.out"
            self.assertIsNone(report._lcurve_final(path))
            path.write_text("step rmse\n", encoding="utf-8")
            self.assertIsNone(report._lcurve_final(path))  # header only
            path.write_text("step rmse\n0 1.5\n100 0.25\n\n", encoding="utf-8")
            self.assertEqual(report._lcurve_final(path), {"step": 100.0, "rmse": 0.25})
            path.write_text("step rmse\n0 1.5\ndone now\n", encoding="utf-8")
            self.assertIsNone(report._lcurve_final(path))

    def test_lcurve_final_with_hash_header_is_shifted_by_one_column(self) -> None:
        # CHARACTERIZATION (suspected bug, med): DeePMD's header starts with "#", which is kept
        # as a column name, so every metric is labelled with its left neighbour's name
        # ("#" -> step, "step" -> rmse_val ...). progress._lcurve_tail strips the "#".
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lcurve.out"
            path.write_text("#  step  rmse_val  lr\n500000  0.11  1e-6\n", encoding="utf-8")
            final = report._lcurve_final(path)
            self.assertEqual(final, {"#": 500000.0, "step": 0.11, "rmse_val": 1e-6})
            self.assertNotIn("lr", final)

    def test_empty_campaign_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = report.build_report(make_campaign(root, name="<x>", description="d & e"))
            html_text = (root / "reports" / "index.html").read_text(encoding="utf-8")
            self.assertIn("No generated campaign artifacts were found yet.", html_text)
            self.assertIn("<title>&lt;x&gt; — InterfaceForge</title>", html_text)
            self.assertIn("d &amp; e", html_text)
            self.assertEqual(
                {key: value for key, value in summary.items() if key.startswith("has_")},
                {"has_plan": False, "has_audit": False, "has_dataset": False,
                 "has_mace": False, "has_deepmd": False},
            )
            sidecar = json.loads((root / "reports" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar, summary)
            self.assertIn("campaign_report", StateStore(root).load()["artifacts"])

    def test_report_sections_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(relative: str, payload: object) -> None:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            write(".interfaceforge/plan.json", {"tasks": [
                {"engine": "vasp", "system": "iface", "stage": "scf", "profile": "p"},
                {"engine": "mace", "stage": "train", "profile": "q"}]})
            write("reports/audit/audit.json", {"runs": [
                {"relative_path": "r1", "ml_mode": "train", "progress_pct": 42.123,
                 "health": "ok", "next_action": "wait"},
                {"relative_path": "r2", "progress_pct": None}]})
            write("datasets/canonical/manifest.json", {
                "strategy": "grouped", "trajectories": 3, "type_map": ["Ni", "O"],
                "frame_counts": {"train": 8, "valid": 1, "test": 1}})
            write("models/mace/training_manifest.json", {"stages": [1, 2]})
            write("models/deepmd/ensemble_manifest.json", {
                "architectures": ["se_e2_a", "dpa2"], "models": [1, 2, 3], "backend": "pytorch"})
            lcurve = root / "models" / "deepmd" / "m0" / "lcurve.out"
            lcurve.parent.mkdir(parents=True)
            lcurve.write_text("step rmse\n100 0.5\n", encoding="utf-8")
            StateStore(root).event("hello", n=1)

            summary = report.build_report(make_campaign(root), output=root / "custom" / "r.html")
            text = (root / "custom" / "r.html").read_text(encoding="utf-8")
            for heading in ("Campaign plan", "VASP and VASP-MLFF health", "Canonical dataset",
                            "Model campaigns", "Completed learning curves", "Provenance log"):
                self.assertIn(f"<h2>{heading}</h2>", text)
            self.assertIn("<td>—</td>", text)  # a task with no system shows an em dash
            self.assertIn("<td>42.1%</td>", text)
            self.assertIn("<td>se_e2_a, dpa2</td><td>3</td><td>pytorch</td>", text)
            self.assertIn("<td>Ni O</td>", text)
            self.assertIn("models/deepmd/m0/lcurve.out", text)
            self.assertTrue(all(summary[key] for key in (
                "has_plan", "has_audit", "has_dataset", "has_mace", "has_deepmd")))
            self.assertEqual(summary["output"], str((root / "custom" / "r.html").resolve()))


OUTCAR = """ vasp.6.3.2 27Jun22 (build Oct 01 2022) complex
   POTCAR:    PAW_PBE Ni 02Aug2007
   TITEL  = PAW_PBE Ni 02Aug2007
   TITEL  = PAW_PBE O 08Apr2002
   TITEL  = PAW_PBE Ni 02Aug2007
   ENCUT  =  520.00 eV
   POTIM  =    1.00
   IVDW   =   12
   NKPTS = 4
 POSITION                                       TOTAL-FORCE (eV/Angst)
 POSITION                                       TOTAL-FORCE (eV/Angst)
"""


def write_leaf(root: Path, *, incar: str = "ENCUT = 520\nIVDW = 12 ! d3\nPOTIM = 1.0\n",
               outcar: str = OUTCAR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "INCAR").write_text(incar, encoding="utf-8")
    (root / "OUTCAR").write_text(outcar, encoding="utf-8")
    (root / "KPOINTS").write_text("k\n0\nGamma\n1 1 1\n", encoding="utf-8")
    return root


class ProvenanceTests(unittest.TestCase):
    def test_sha256_file_streams_in_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "f"
            data = b"0123456789" * 100
            path.write_bytes(data)
            self.assertEqual(vasp_provenance.sha256_file(path, chunk_size=7), hashlib.sha256(data).hexdigest())

    def test_equivalent_value(self) -> None:
        equivalent = vasp_provenance._equivalent_value
        self.assertTrue(equivalent("520", "520.00"))
        self.assertTrue(equivalent("1e-3", "0.001"))
        self.assertFalse(equivalent("520", "500"))
        self.assertTrue(equivalent(".TRUE.", " .true. "))  # non-numeric: case/space-insensitive
        self.assertFalse(equivalent("Normal", "Fast"))
        self.assertTrue(equivalent(str(math.pi), "3.141592653589793"))

    def test_outcar_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "OUTCAR"
            path.write_text(OUTCAR, encoding="utf-8")
            info = vasp_provenance._outcar_fingerprint(path, tracked_tags=["ENCUT", "IVDW", "POTIM", "ABSENT"])
            self.assertEqual(info["outcar_sha256"], hashlib.sha256(OUTCAR.encode()).hexdigest())
            self.assertEqual(info["ionic_frames_detected"], 2)
            self.assertEqual(info["vasp_version"], "vasp.6.3.2")
            self.assertEqual(info["potcar_titles"], ["PAW_PBE Ni 02Aug2007", "PAW_PBE O 08Apr2002"])
            self.assertEqual(info["nkpts"], "4")
            self.assertEqual(info["outcar_executed_tags"], {"ENCUT": "520.00", "POTIM": "1.00", "IVDW": "12"})

    def test_outcar_fingerprint_scans_tags_only_in_the_header_but_counts_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "OUTCAR"
            body = "filler\n" * 5 + "   ENCUT  =  999.0\n" + " POSITION TOTAL-FORCE\n"
            path.write_text(body, encoding="utf-8")
            info = vasp_provenance._outcar_fingerprint(path, tracked_tags=["ENCUT"], header_lines=3)
            self.assertEqual(info["outcar_executed_tags"], {})
            self.assertEqual(info["ionic_frames_detected"], 1)
            self.assertEqual(info["outcar_sha256"], hashlib.sha256(body.encode()).hexdigest())

    def test_build_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            leaf = write_leaf(Path(temporary) / "leaf")
            record = vasp_provenance.build_vasp_reference_record(
                source_leaf=leaf, source_outcar=leaf / "OUTCAR", staged_leaf="stage/leaf",
                included_files=["INCAR", "OUTCAR", "KPOINTS", "POTCAR"],
            )
            self.assertEqual(sorted(record["file_sha256"]), ["INCAR", "KPOINTS", "OUTCAR"])  # no POTCAR on disk
            self.assertEqual(record["file_sha256"]["OUTCAR"], record["outcar_sha256"])
            self.assertEqual(record["effective_required_settings"], {"ENCUT": "520.00", "IVDW": "12", "POTIM": "1.00"})
            self.assertEqual(record["incar_tags"], {"ENCUT": "520", "IVDW": "12", "POTIM": "1.0"})
            self.assertEqual(record["missing_required_settings"], [])
            self.assertEqual(record["required_incar_tags"], ["ENCUT", "IVDW", "POTIM"])

    def test_build_record_prefers_outcar_values_and_reports_missing_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            leaf = write_leaf(Path(temporary) / "leaf", incar="ENCUT = 400\n", outcar=" vasp.6.4\n")
            record = vasp_provenance.build_vasp_reference_record(
                source_leaf=leaf, source_outcar=leaf / "OUTCAR", staged_leaf="s",
                included_files=["INCAR"], required_incar_tags=["encut", "NELM"],
            )
            self.assertEqual(record["required_incar_tags"], ["ENCUT", "NELM"])
            self.assertEqual(record["effective_required_settings"], {"ENCUT": "400", "NELM": ""})
            self.assertEqual(record["missing_required_settings"], ["NELM"])

    def test_build_record_honours_resolved_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = write_leaf(root / "leaf")
            shared = root / "shared_incar"
            shared.write_text("ENCUT = 300\n", encoding="utf-8")
            record = vasp_provenance.build_vasp_reference_record(
                source_leaf=leaf, source_outcar=leaf / "OUTCAR", staged_leaf="s",
                included_files=["INCAR"], file_paths={"INCAR": shared},
            )
            self.assertEqual(record["incar_tags"], {"ENCUT": "300"})
            self.assertEqual(record["file_sha256"]["INCAR"], hashlib.sha256(b"ENCUT = 300\n").hexdigest())
            self.assertEqual(record["resolved_input_paths"], {"INCAR": str(shared)})

    def records(self, root: Path, count: int = 2, **leaf_kwargs):
        built = []
        for index in range(count):
            leaf = write_leaf(root / f"leaf{index}", **leaf_kwargs)
            built.append(vasp_provenance.build_vasp_reference_record(
                source_leaf=leaf, source_outcar=leaf / "OUTCAR", staged_leaf=f"s{index}",
                included_files=["INCAR", "OUTCAR", "KPOINTS"],
            ))
        return built

    def test_audit_of_consistent_records_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit = vasp_provenance.audit_vasp_reference_records(self.records(Path(temporary)))
            self.assertEqual(audit["status"], "OK")
            self.assertEqual(audit["problems"], [])
            self.assertEqual(audit["vasp_versions"], ["vasp.6.3.2"])
            self.assertTrue(audit["exact_incar_files_identical"])
            self.assertEqual(audit["ionic_frame_counts"], {"minimum": 2, "maximum": 2, "unique": [2]})
            self.assertEqual(audit["file_hash_coverage"], {"INCAR": 2, "KPOINTS": 2, "OUTCAR": 2})
            self.assertEqual(audit["outcar_echo_coverage"], {"ENCUT": 2, "IVDW": 2, "POTIM": 2})
            self.assertEqual(audit["potcar_title_sets"], ["PAW_PBE Ni 02Aug2007 | PAW_PBE O 08Apr2002"])

    def test_audit_flags_problems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            good = self.records(Path(temporary) / "a", count=1)[0]
            mismatch = self.records(Path(temporary) / "b", count=1, incar="ENCUT = 400\nIVDW = 12\nPOTIM = 1.0\n")[0]
            empty = {"staged_leaf": "bare", "missing_required_settings": ["ENCUT"]}
            audit = vasp_provenance.audit_vasp_reference_records([good, mismatch, empty])
            self.assertEqual(audit["status"], "FAILED")
            issues = [issue for problem in audit["problems"] for issue in problem["issues"]]
            self.assertIn("INCAR/OUTCAR mismatch for ENCUT: input=400, executed=520.00", issues)
            self.assertIn("missing ENCUT from both INCAR and OUTCAR", issues)
            self.assertIn("no ionic POSITION/TOTAL-FORCE frames detected in OUTCAR", issues)
            self.assertIn("missing both KPOINTS hash and OUTCAR NKPTS provenance", issues)
            self.assertIn("missing both POTCAR hash and OUTCAR TITEL provenance", issues)
            self.assertTrue(any(issue.startswith("inconsistent ENCUT:") for issue in issues))
            self.assertTrue(any(issue.startswith("mixed or unknown VASP versions") for issue in issues))
            self.assertFalse(audit["exact_incar_files_identical"])
            self.assertIn("ENCUT", audit["differing_incar_tags"])

    def test_audit_of_no_records_reports_unknown_version(self) -> None:
        audit = vasp_provenance.audit_vasp_reference_records([])
        self.assertEqual(audit["status"], "FAILED")
        self.assertEqual(audit["vasp_versions"], [])
        self.assertEqual(audit["ionic_frame_counts"]["minimum"], 0)

    def test_write_provenance_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self.records(root)
            audit = vasp_provenance.audit_vasp_reference_records(records)
            written = vasp_provenance.write_vasp_reference_provenance(records, audit, root / "out" / "prov")
            self.assertEqual(sorted(written), ["audit", "csv", "records"])
            payload = json.loads(Path(written["records"]).read_text())
            self.assertEqual((payload["schema_version"], len(payload["records"])), (1, 2))
            with Path(written["csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["staged_leaf"] for row in rows], ["s0", "s1"])
            self.assertEqual(
                (rows[0]["ENCUT"], rows[0]["incar_ENCUT"], rows[0]["outcar_ENCUT"]),
                ("520.00", "520", "520.00"),
            )
            self.assertEqual(rows[0]["potcar_titles"], "PAW_PBE Ni 02Aug2007 | PAW_PBE O 08Apr2002")
            self.assertEqual(rows[0]["potcar_sha256"], "")
            self.assertEqual(json.loads(Path(written["audit"]).read_text())["status"], "OK")

    def test_write_provenance_with_no_records_raises_index_error(self) -> None:
        # CHARACTERIZATION (suspected bug, low): ``csv_rows[0]`` fails on an empty record list.
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(IndexError):
                vasp_provenance.write_vasp_reference_provenance([], {}, Path(temporary) / "p")


if __name__ == "__main__":
    unittest.main()
