from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from interfaceforge.committee import collect_committee, verify_committee_bundle
from interfaceforge.errors import SafetyError


def write_committee(root: Path, seeds: tuple[int, ...] = (0, 211, 307, 419)) -> Path:
    source = root / "mace_committee"
    for seed in seeds:
        run = source / f"seed_{seed}"
        for name in ("results", "mace_model", "checkpoints", "logs"):
            (run / name).mkdir(parents=True, exist_ok=True)
        (run / "mace_model" / "TiN_SiN_mace_stagetwo.model").write_bytes(
            f"trained-model-seed-{seed}".encode("utf-8")
        )
    return source


class CommitteeCollectorTests(unittest.TestCase):
    def test_collects_sorted_distinct_models_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_committee(root)
            training = root / "train.extxyz"
            training.write_text("training-data\n", encoding="utf-8")

            result = collect_committee(
                source,
                root / "stored_models/tin_sin_v1",
                training_data=[training],
                training_data_output=root / "stored_data/tin_sin_training_v1.zip",
                label="TiN/SiN MACE committee v1",
            )

            bundle = Path(result["bundle"])
            archive = Path(result["archive"])
            self.assertTrue(archive.is_file())
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([member["seed"] for member in manifest["members"]], [0, 211, 307, 419])
            self.assertEqual(manifest["model_count"], 4)
            self.assertEqual(len({member["sha256"] for member in manifest["members"]}), 4)
            self.assertEqual(len(manifest["training_data"]), 1)
            self.assertEqual(
                (bundle / "committee-models.txt").read_text(encoding="utf-8").splitlines(),
                [
                    "models/seed_0.model",
                    "models/seed_211.model",
                    "models/seed_307.model",
                    "models/seed_419.model",
                ],
            )
            for seed in (0, 211, 307, 419):
                self.assertTrue(
                    (source / f"seed_{seed}/mace_model/TiN_SiN_mace_stagetwo.model").is_file()
                )
            self.assertTrue(verify_committee_bundle(bundle)["valid"])
            zip_result = verify_committee_bundle(archive)
            self.assertTrue(zip_result["valid"])
            self.assertEqual(zip_result["kind"], "zip")
            with zipfile.ZipFile(archive) as handle:
                self.assertIn("tin_sin_v1/models/seed_0.model", handle.namelist())
                self.assertFalse(any(name.endswith(".extxyz") for name in handle.namelist()))
            data_archive = Path(result["training_data_archive"])
            data_result = verify_committee_bundle(data_archive)
            self.assertTrue(data_result["valid"])
            self.assertEqual(data_result["kind"], "training_data_zip")
            with zipfile.ZipFile(data_archive) as handle:
                self.assertIn("tin_sin_training_v1/data/train.extxyz", handle.namelist())

    def test_zip_output_name_creates_directory_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_committee(root)
            result = collect_committee(source, root / "stored_models/tin_sin_v1.zip")
            self.assertEqual(Path(result["bundle"]).name, "tin_sin_v1")
            self.assertEqual(Path(result["archive"]).name, "tin_sin_v1.zip")
            self.assertTrue(Path(result["bundle"]).is_dir())
            self.assertTrue(Path(result["archive"]).is_file())

    def test_rejects_incomplete_or_duplicate_committee(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = write_committee(root, seeds=(0, 211, 307))
            with self.assertRaisesRegex(SafetyError, "Expected 4"):
                collect_committee(incomplete, root / "incomplete-bundle")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_committee(root)
            original = source / "seed_0/mace_model/TiN_SiN_mace_stagetwo.model"
            duplicate = source / "seed_211/mace_model/TiN_SiN_mace_stagetwo.model"
            duplicate.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(SafetyError, "Duplicate committee model"):
                collect_committee(source, root / "duplicate-bundle")

    def test_bundle_is_immutable_and_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_committee(root)
            bundle = root / "bundle"
            result = collect_committee(source, bundle)
            with self.assertRaisesRegex(SafetyError, "already exists"):
                collect_committee(source, bundle)

            (bundle / "models/seed_307.model").write_bytes(b"modified")
            with self.assertRaisesRegex(SafetyError, "checksum mismatch"):
                verify_committee_bundle(bundle)

            Path(result["archive"]).write_bytes(b"not-a-zip")
            with self.assertRaisesRegex(SafetyError, "Invalid committee ZIP"):
                verify_committee_bundle(result["archive"])


if __name__ == "__main__":
    unittest.main()
