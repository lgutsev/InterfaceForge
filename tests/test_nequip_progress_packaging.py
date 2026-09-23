"""NequIP in mlip-progress (read-only) and in committee / Hugging Face / campaign packaging."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml
from test_nequip_backend import install_fakes, make_campaign

from interfaceforge.committee import collect_committee, verify_committee_bundle
from interfaceforge.config import load_campaign
from interfaceforge.errors import SafetyError
from interfaceforge.nequip import evaluate_nequip_committee, generate_nequip_training
from interfaceforge.packaging import pack_campaign, pack_huggingface
from interfaceforge.progress import mlip_progress, render


class _TrainedCommittee(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.campaign_dir = make_campaign(
            self.root, nequip={"device": "cpu", "profile": "nequip_cpu", "seeds": [11, 23]}
        )
        self.manifest = generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"))
        self.nequip_root = Path(self.manifest["root"])
        self.env = install_fakes(self.root)

    def train(self, *seeds: int) -> None:
        for seed in seeds:
            result = subprocess.run(
                ["bash", str(self.nequip_root / f"seed_{seed}" / "train_member.sh")],
                env=self.env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def evaluate(self) -> None:
        for index, member in enumerate(self.manifest["members"]):
            subprocess.run(
                [
                    sys.executable,
                    str(self.nequip_root / "evaluate_nequip.py"),
                    "--model",
                    str(Path(member["directory"]) / "final" / "model.nequip.pt2"),
                    "--frames",
                    self.manifest["evaluation"]["test_file"],
                    "--device",
                    "cpu",
                    "--label",
                    f"model_{index:03d}",
                    "--seed",
                    str(member["seed"]),
                    "--output",
                    str(Path(member["directory"]) / "evaluation"),
                ],
                env=self.env,
                check=True,
                capture_output=True,
            )
        evaluate_nequip_committee(self.nequip_root)


class ProgressTests(_TrainedCommittee):
    def test_nequip_members_appear_and_progress_is_read_only(self) -> None:
        self.train(11)
        before = {path: path.stat().st_mtime_ns for path in self.campaign_dir.rglob("*") if path.is_file()}
        payload = mlip_progress(self.campaign_dir)
        after = {path: path.stat().st_mtime_ns for path in self.campaign_dir.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        committee = payload["nequip_committees"][0]
        states = {member["seed"]: member for member in committee["members"]}
        self.assertEqual(states[11]["state"], "complete")
        self.assertEqual(states[11]["epoch"], 3)
        self.assertTrue(states[11]["best_checkpoint"] and states[11]["compiled_model"])
        self.assertEqual(states[23]["state"], "not-started")
        self.assertFalse(committee["committee_evaluation"])
        text = render(payload)
        self.assertIn("NequIP training", text)
        self.assertIn("seed 11", text)
        self.assertIn("committee evaluation: no", text)
        self.assertEqual(mlip_progress(self.root / "elsewhere")["nequip_committees"], [])


class PackagingTests(_TrainedCommittee):
    def test_committee_bundle_preserves_provenance_and_verifies(self) -> None:
        self.train(11, 23)
        self.evaluate()
        result = collect_committee(
            self.nequip_root, self.root / "stored" / "nio_nequip_v1", engine="nequip", expected_members=2
        )
        bundle = Path(result["bundle"])
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual((manifest["engine"], manifest["architecture"]), ("nequip", "nequip"))
        self.assertEqual(manifest["type_map"], ["C", "H", "N", "Ni", "O", "P"])
        self.assertEqual(manifest["dataset"]["split_hash"], self.manifest["dataset"]["identity"]["split_hash"])
        self.assertIn("r_max", manifest["hyperparameters"])
        self.assertIn("learning_rate", manifest["defaults_applied"])
        self.assertTrue(manifest["training_interfaceforge_commit"]["source"])
        self.assertIsNotNone(manifest["committee_evaluation"]["ensemble"])
        member = manifest["members"][0]
        self.assertEqual(member["stored_model"], "models/seed_11.nequip.zip")
        self.assertEqual({extra["kind"] for extra in member["extra_files"]}, {"compiled_model", "config", "provenance"})
        provenance = json.loads((bundle / "provenance" / "seed_11.json").read_text())
        self.assertEqual(provenance["seed"], 11)
        self.assertTrue(provenance["config_sha256"])
        self.assertEqual(provenance["versions"]["nequip"], "0.0-stub")
        self.assertIsNotNone(provenance["committee_evaluation_metrics"])
        names = {path.name for path in bundle.rglob("*")}
        self.assertFalse(any(name.endswith(".ckpt") for name in names))
        self.assertTrue(verify_committee_bundle(bundle)["valid"])
        self.assertTrue(verify_committee_bundle(result["archive"])["valid"])

        (bundle / "configs" / "seed_23.config.yaml").write_text("tampered\n")
        with self.assertRaisesRegex(SafetyError, "extra file"):
            verify_committee_bundle(bundle)

    def test_checkpoints_only_on_request_and_incomplete_members_refused(self) -> None:
        self.train(11)
        with self.assertRaisesRegex(SafetyError, "model.nequip.zip|not complete"):
            collect_committee(self.nequip_root, self.root / "bad", engine="nequip", expected_members=2)
        self.train(23)
        result = collect_committee(
            self.nequip_root, self.root / "with_ckpt", engine="nequip", expected_members=2, include_checkpoints=True
        )
        with zipfile.ZipFile(result["archive"]) as handle:
            self.assertIn("with_ckpt/checkpoints/seed_11.best.ckpt", handle.namelist())

    def test_huggingface_package_and_campaign_packaging(self) -> None:
        self.train(11, 23)
        self.evaluate()
        bundle = collect_committee(self.nequip_root, self.root / "stored" / "b", engine="nequip", expected_members=2)
        hf = pack_huggingface(
            bundle["bundle"], self.root / "hf", metrics_path=self.nequip_root / "evaluation" / "summary.json"
        )
        card = (Path(hf["output"]) / "README.md").read_text()
        self.assertIn("library_name: nequip", card)
        self.assertIn("not a calibrated", card)
        self.assertIn("nequip-compile", card)
        self.assertIn("Split hash", card)
        self.assertTrue(hf["has_metrics"])
        provenance = json.loads((Path(hf["output"]) / "interfaceforge_manifest.json").read_text())
        self.assertEqual(provenance["dataset"]["split_hash"], self.manifest["dataset"]["identity"]["split_hash"])
        self.assertTrue((Path(hf["output"]) / "configs" / "seed_11.config.yaml").is_file())

        campaign = load_campaign(self.campaign_dir / "campaign.yaml")
        payload = pack_campaign(campaign, output_root=self.root / "packaged", include_dataset_archive=False)
        components = {entry["component"] for entry in payload["committees"]}
        self.assertIn("nequip", components)
        self.assertEqual(payload["errors"], [])

    def test_enabled_but_missing_nequip_committee_is_reported(self) -> None:
        data = yaml.safe_load((self.campaign_dir / "campaign.yaml").read_text())
        data["models"]["nequip"]["output_dir"] = "models/nequip_elsewhere"
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        payload = pack_campaign(
            load_campaign(self.campaign_dir / "campaign.yaml"),
            output_root=self.root / "p2",
            nequip_root=self.root / "no_such_root",
            include_dataset_archive=False,
        )
        self.assertIn("nequip_committee", {entry["step"] for entry in payload["skipped"]})


if __name__ == "__main__":
    unittest.main()
