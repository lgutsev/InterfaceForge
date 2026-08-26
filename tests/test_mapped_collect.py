from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from interfaceforge.leaf_audit import audit_leaf_manifests, write_leaf_audit
from interfaceforge.mapped_collect import (
    discover_mapped_leaves,
    load_mapped_config,
    run_mapped_collection,
    stage_mapped_leaves,
)


class MappedCollectionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        source = root / "source"
        for leaf in (source / "N_Term" / "x0.0", source / "N_Term" / "x0.5"):
            leaf.mkdir(parents=True)
            (leaf / "OUTCAR").write_text("vasp output\n", encoding="utf-8")
            (leaf / "INCAR").write_text("TEBEG = 300\n", encoding="utf-8")
            # A daughter archive must not be reproduced in the clean staging tree.
            archive = leaf / "restart_archive_1"
            archive.mkdir()
            (archive / "OUTCAR").write_text("old output\n", encoding="utf-8")
        config = {
            "schema_version": 1,
            "campaign_root": "${TEST_CAMPAIGN_ROOT}",
            "staging_root": "reference_runs",
            "initialize_campaign": False,
            "sources": [{"source": str(source), "target": "interface/300K/Real"}],
            "collection": {
                "ratios": [0.8, 0.1, 0.1],
                "seed": 7,
                "stride": 5,
                "type_map": ["Si", "N", "Ti", "O"],
            },
        }
        path = root / "mapped.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_dry_run_resolves_mapping_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            with patch.dict(os.environ, {"TEST_CAMPAIGN_ROOT": str(campaign)}):
                payload = run_mapped_collection(self._fixture(root))

            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(len(payload["leaves"]), 2)
            self.assertFalse(campaign.exists())

    def test_staging_hardlinks_only_selected_direct_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            with patch.dict(os.environ, {"TEST_CAMPAIGN_ROOT": str(campaign)}):
                config = load_mapped_config(self._fixture(root))
                leaves = discover_mapped_leaves(config)
                payload = stage_mapped_leaves(config, leaves)

            self.assertEqual(payload["leaves"], 2)
            staged = campaign / "reference_runs" / "interface" / "300K" / "Real"
            outcars = sorted(staged.rglob("OUTCAR"))
            self.assertEqual(len(outcars), 2)
            self.assertFalse(any("archive" in str(path) for path in outcars))
            self.assertTrue(os.path.samefile(outcars[0], leaves[0]["source_outcar"]))


class LeafAuditTests(unittest.TestCase):
    @staticmethod
    def _manifest(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("relative_leaf", "split", "frames", "status", "detail"),
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_synchronized_manifests_create_visual_report(self) -> None:
        rows = [
            {
                "relative_leaf": "interface/300K/Real/N_Term/a",
                "split": "train",
                "frames": 10,
                "status": "OK",
                "detail": "",
            },
            {
                "relative_leaf": "interface/450K/Real/Ti_Term/b",
                "split": "valid",
                "frames": 8,
                "status": "OK",
                "detail": "",
            },
            {
                "relative_leaf": "bulk/TiO-Bulk-Real_450K",
                "split": "test",
                "frames": 6,
                "status": "OK",
                "detail": "",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mace = root / "mace.csv"
            deepmd = root / "deepmd.csv"
            self._manifest(mace, rows)
            self._manifest(deepmd, rows)

            report = audit_leaf_manifests(mace, deepmd)
            outputs = write_leaf_audit(report, root / "audit")

            self.assertEqual(report["status"], "OK")
            self.assertEqual(report["split_frames"], {"train": 10, "valid": 8, "test": 6})
            self.assertTrue(Path(outputs["svg"]).is_file())
            self.assertIn("<svg", Path(outputs["svg"]).read_text(encoding="utf-8"))

    def test_split_mismatch_fails(self) -> None:
        mace_rows = [
            {"relative_leaf": "bulk/A", "split": "train", "frames": 5, "status": "OK", "detail": ""}
        ]
        deepmd_rows = [
            {"relative_leaf": "bulk/A", "split": "test", "frames": 5, "status": "OK", "detail": ""}
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mace = root / "mace.csv"
            deepmd = root / "deepmd.csv"
            self._manifest(mace, mace_rows)
            self._manifest(deepmd, deepmd_rows)
            report = audit_leaf_manifests(mace, deepmd)

            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("missing" in issue for problem in report["problems"] for issue in problem["issues"]))


    def test_same_leaf_can_appear_in_every_synchronized_split(self) -> None:
        rows = [
            {
                "relative_leaf": "interface/300K/Real/N_Term/a",
                "split": split,
                "frames": frames,
                "status": "OK",
                "detail": "",
            }
            for split, frames in (("train", 80), ("valid", 10), ("test", 10))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mace = root / "mace.csv"
            deepmd = root / "deepmd.csv"
            self._manifest(mace, rows)
            self._manifest(deepmd, rows)

            report = audit_leaf_manifests(mace, deepmd)

            self.assertEqual(report["status"], "OK")
            self.assertEqual(report["leaves"], 1)
            self.assertEqual(report["split_leaves"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(
                report["split_frames"],
                {"train": 80, "valid": 10, "test": 10},
            )


if __name__ == "__main__":
    unittest.main()
