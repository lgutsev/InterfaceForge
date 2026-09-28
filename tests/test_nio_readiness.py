"""Pre-GPU NiO readiness audit: answers, blocking items, and three-backend consumption."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml
from nio_fixture import build_nio_tree

from interfaceforge.cli import main as cli_main
from interfaceforge.config import load_campaign
from interfaceforge.nio_dataset import ExportConfig, export_dataset
from interfaceforge.nio_readiness import backend_consumption, readiness_audit

REPO = Path(__file__).resolve().parents[1]


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.head = build_nio_tree(
            self.root / "NiO_head",
            special={"Step2_600K/NiO_m110_Big_U46_OH75_scattered_capped": {"truncate_last": True}},
        )
        self.campaign = self.root / "campaign"
        (self.campaign / "profiles").mkdir(parents=True)
        shutil.copy(REPO / "profiles" / "loni.yaml", self.campaign / "profiles" / "loni.yaml")
        (self.campaign / "s.vasp").write_text("x\n")
        (self.campaign / "campaign.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "project": {"name": "nio"},
                    "profile": "profiles/loni.yaml",
                    "systems": [{"id": "nio", "kind": "surface", "structure": "s.vasp"}],
                    "models": {
                        "mace": {"enabled": True},
                        "deepmd": {"enabled": True, "backend": "pytorch", "architectures": ["dpa2"]},
                        "nequip": {"enabled": True, "dataset": "datasets/canonical", "r_max": 5.0},
                    },
                }
            )
        )

    def test_audit_before_export_answers_questions_but_is_not_ready(self) -> None:
        payload = readiness_audit([self.head], ExportConfig(), output=self.root / "audit")
        answers = payload["answers"]
        self.assertEqual(answers["discoverable_trajectories"]["total"], 24)
        self.assertEqual(sum(answers["frames_per_split"].values()), payload["summary"]["frames_selected"])
        self.assertFalse(answers["leakage"]["leakage_detected"])
        self.assertIn("OH50/NiO_m110_Big_U46_OH50_clustered_capped", answers["usable_frames_per_system_temperature"])
        problem_ids = {row["trajectory_id"] for row in answers["problem_runs"]}
        self.assertIn("NiO_head/Step2_600K/OH75/NiO_m110_Big_U46_OH75_scattered_capped", problem_ids)
        self.assertFalse(payload["ready_for_gpu_smoke_tests"])
        self.assertTrue(any("not exported" in item for item in payload["blocking_items"]))
        markdown = (self.root / "audit" / "readiness.md").read_text()
        for heading in (
            "## 1. Discoverable trajectories",
            "## 3. Incomplete or problematic runs",
            "## 5. Leakage",
            "## 6. Backend consumption",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("does not validate", markdown)
        self.assertTrue(json.loads((self.root / "audit" / "readiness.json").read_text())["trajectories"])

    def test_exported_dataset_and_campaign_make_all_backends_consistent(self) -> None:
        export_dataset([self.head], self.campaign / "datasets" / "canonical", ExportConfig())
        campaign = load_campaign(self.campaign / "campaign.yaml")
        payload = readiness_audit(
            [self.head],
            ExportConfig(),
            output=self.root / "audit",
            campaign=campaign,
            dataset=self.campaign / "datasets" / "canonical",
        )
        consumption = payload["answers"]["backend_consumption"]
        self.assertTrue(consumption["all_enabled_backends_share_dataset_and_split"], consumption)
        self.assertEqual(set(consumption["backends"]), {"mace", "deepmd", "nequip"})
        self.assertTrue(payload["answers"]["dataset_verification"]["valid"])
        self.assertTrue(payload["ready_for_gpu_smoke_tests"], payload["blocking_items"])

        # MACE pointed at a modified copy is caught.
        copy = self.campaign / "other" / "train.extxyz"
        copy.parent.mkdir()
        copy.write_text((self.campaign / "datasets" / "canonical" / "train.extxyz").read_text() + "\n")
        data = yaml.safe_load((self.campaign / "campaign.yaml").read_text())
        data["models"]["mace"]["train_file"] = "other/train.extxyz"
        (self.campaign / "campaign.yaml").write_text(yaml.safe_dump(data))
        check = backend_consumption(
            load_campaign(self.campaign / "campaign.yaml"), self.campaign / "datasets" / "canonical"
        )
        self.assertFalse(check["backends"]["mace"]["same_dataset"])
        self.assertFalse(check["all_enabled_backends_share_dataset_and_split"])

    def test_cli_readiness(self) -> None:
        dataset = self.root / "dataset"
        with redirect_stdout(StringIO()):
            cli_main(["dataset", "export", str(self.head), "--output", str(dataset)])
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = cli_main(
                [
                    "dataset",
                    "readiness",
                    str(self.head),
                    "--output",
                    str(self.root / "ready"),
                    "--dataset",
                    str(dataset),
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(buffer.getvalue())
        self.assertTrue(summary["ready_for_gpu_smoke_tests"], summary)
        self.assertTrue(Path(summary["outputs"]["markdown"]).is_file())


if __name__ == "__main__":
    unittest.main()
