from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from test_committee import write_committee

from interfaceforge.cli import build_parser
from interfaceforge.committee import collect_committee, verify_committee_bundle
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.packaging import (
    pack_dataset_archive,
    pack_huggingface,
    verify_package,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def write_canonical_dataset(root: Path) -> Path:
    dataset = root / "datasets" / "canonical"
    (dataset / "deepmd").mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (dataset / f"{split}.extxyz").write_text(
            "1\nProperties=species:S:1:pos:R:3 REF_energy=-1.0\nSi 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        system = dataset / "deepmd" / split / "system"
        set_dir = system / "set.000"
        set_dir.mkdir(parents=True)
        (system / "type.raw").write_text("0\n1\n", encoding="utf-8")
        (system / "type_map.raw").write_text("Si\nN\n", encoding="utf-8")
        np.save(set_dir / "coord.npy", np.zeros((1, 6)))
        np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
        np.save(set_dir / "energy.npy", np.zeros((1, 1)))
        np.save(set_dir / "force.npy", np.zeros((1, 6)))
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy": "grouped",
                "ratios": [0.8, 0.1, 0.1],
                "stride": 5,
                "type_map": ["Si", "N"],
                "frame_counts": {"train": 8, "valid": 1, "test": 1},
                "deepmd": {split: str(dataset / "deepmd" / split) for split in ("train", "valid", "test")},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "manifest.csv").write_text("run_id,frames_retained\nrun_a,10\n", encoding="utf-8")
    (dataset / "frames.csv").write_text("split,run_id,energy_ev\ntrain,run_a,-1.0\n", encoding="utf-8")
    return dataset


def write_deepmd_committee(
    root: Path,
    *,
    architecture: str = "dpa2",
    members: int = 4,
    finetune: bool = False,
) -> Path:
    deepmd = root / "models" / "deepmd"
    arch_dir = deepmd / architecture
    ensemble_models = []
    for index in range(members):
        model_dir = arch_dir / f"model_{index:03d}"
        model_dir.mkdir(parents=True)
        (model_dir / "frozen_model.pth").write_bytes(f"frozen-{architecture}-{index}".encode())
        (model_dir / "model.ckpt.pt").write_bytes(b"ckpt")
        (model_dir / "lcurve.out").write_text("# step\n100 0.1\n", encoding="utf-8")
        (model_dir / "input.json").write_text(
            json.dumps(
                {"model": {"type_map": ["Si", "N"]}, "training": {"numb_steps": 100}}
            ),
            encoding="utf-8",
        )
        ensemble_models.append(
            {"architecture": architecture, "index": index, "seed": 11 + index * 12}
        )
    manifest = {
        "schema_version": 1,
        "engine": "deepmd",
        "backend": "pytorch",
        "architectures": [architecture],
        "type_map": ["Si", "N"],
        "campaign": "sintin",
        "models": ensemble_models,
    }
    if finetune:
        manifest["finetune"] = {architecture: {"pretrained": "/foundation/dpa2_openlam.pt"}}
    (deepmd / "ensemble_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return arch_dir


# --------------------------------------------------------------------------- #
# dataset archive
# --------------------------------------------------------------------------- #
class DatasetArchiveTests(unittest.TestCase):
    def test_archive_roundtrips_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_canonical_dataset(root)
            result = pack_dataset_archive(dataset, root / "backup" / "sintin_v1.zip", label="SiN/TiN v1")

            archive = Path(result["archive"])
            self.assertTrue(archive.is_file())
            self.assertEqual(result["artifact_type"], "mlip_dataset_archive")
            self.assertTrue(result["include_extxyz"])

            report = verify_package(archive)
            self.assertTrue(report["valid"])
            self.assertEqual(report["kind"], "zip")

            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
            self.assertIn("sintin_v1/data/train.extxyz", names)
            self.assertIn("sintin_v1/data/deepmd/train/system/set.000/coord.npy", names)
            self.assertIn("sintin_v1/data/manifest.json", names)
            self.assertIn("sintin_v1/README.md", names)

    def test_no_extxyz_option_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_canonical_dataset(root)
            result = pack_dataset_archive(
                dataset, root / "backup.zip", include_extxyz=False
            )
            with zipfile.ZipFile(Path(result["archive"])) as handle:
                names = handle.namelist()
            self.assertFalse(any(name.endswith(".extxyz") for name in names))

            extracted = root / "restored"
            with zipfile.ZipFile(Path(result["archive"])) as handle:
                handle.extractall(extracted)
            top = extracted / "backup"
            self.assertTrue(verify_package(top)["valid"])
            (top / "data" / "manifest.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "Checksum mismatch"):
                verify_package(top)

    def test_refuses_overwrite_and_rejects_non_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_canonical_dataset(root)
            out = root / "sintin.zip"
            pack_dataset_archive(dataset, out)
            with self.assertRaisesRegex(SafetyError, "already exists"):
                pack_dataset_archive(dataset, out)
            pack_dataset_archive(dataset, out, force=True)

            (root / "not-a-dataset").mkdir()
            with self.assertRaisesRegex(ConfigurationError, "no manifest.json"):
                pack_dataset_archive(root / "not-a-dataset", root / "x.zip")


# --------------------------------------------------------------------------- #
# DeePMD committee collection
# --------------------------------------------------------------------------- #
class DeepMDCommitteeCollectTests(unittest.TestCase):
    def test_collects_frozen_models_with_ensemble_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_deepmd_committee(root)
            result = collect_committee(
                source, root / "stored" / "dpa2_v1", engine="deepmd", label="SiN/TiN DPA-2 v1"
            )
            bundle = Path(result["bundle"])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["engine"], "deepmd")
            self.assertEqual(manifest["architecture"], "dpa2")
            self.assertEqual(manifest["backend"], "pytorch")
            self.assertEqual(manifest["type_map"], ["Si", "N"])
            self.assertEqual([m["seed"] for m in manifest["members"]], [11, 23, 35, 47])
            self.assertEqual(
                [m["stored_model"] for m in manifest["members"]],
                [f"models/model_{i:03d}.pth" for i in range(4)],
            )
            self.assertTrue((bundle / "models" / "model_000.pth").is_file())
            self.assertTrue(verify_committee_bundle(bundle)["valid"])
            self.assertTrue(verify_committee_bundle(Path(result["archive"]))["valid"])

    def test_missing_frozen_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_deepmd_committee(root)
            (source / "model_002" / "frozen_model.pth").unlink()
            with self.assertRaisesRegex(SafetyError, "no non-empty frozen model"):
                collect_committee(source, root / "stored" / "dpa2_v1", engine="deepmd")

    def test_collects_without_ensemble_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_deepmd_committee(root)
            (source.parent / "ensemble_manifest.json").unlink()
            bundle = Path(
                collect_committee(source, root / "stored" / "v1", engine="deepmd")["bundle"]
            )
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["architecture"], "dpa2")
            self.assertEqual(manifest["backend"], "pytorch")
            self.assertEqual(manifest["type_map"], ["Si", "N"])
            self.assertEqual([m["seed"] for m in manifest["members"]], [0, 1, 2, 3])
            self.assertTrue(verify_committee_bundle(bundle)["valid"])

    def test_duplicate_member_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_deepmd_committee(root)
            (source / "model_001" / "frozen_model.pth").write_bytes(
                (source / "model_000" / "frozen_model.pth").read_bytes()
            )
            with self.assertRaisesRegex(SafetyError, "Duplicate committee model"):
                collect_committee(source, root / "stored" / "dpa2_v1", engine="deepmd")


# --------------------------------------------------------------------------- #
# Hugging Face packaging
# --------------------------------------------------------------------------- #
class HuggingFacePackageTests(unittest.TestCase):
    def _mace_bundle(self, root: Path) -> Path:
        source = write_committee(root)
        result = collect_committee(source, root / "stored" / "mace_v1", expected_members=4)
        return Path(result["bundle"])

    def test_mace_package_is_upload_ready_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._mace_bundle(root)
            result = pack_huggingface(
                bundle, root / "hf" / "mace_v1", repo_id="myorg/tinsin-mace", make_zip=True
            )
            out = Path(result["output"])
            readme = (out / "README.md").read_text(encoding="utf-8")
            self.assertTrue(readme.startswith("---\n"))
            self.assertIn("library_name: mace", readme)
            self.assertIn("hf upload", (out / "UPLOAD.md").read_text(encoding="utf-8"))
            for name in (".gitattributes", "checksums.sha256", "interfaceforge_manifest.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertTrue((out / "models" / "seed_0.model").is_file())
            self.assertTrue(verify_package(out)["valid"])
            self.assertTrue(verify_package(Path(result["archive"]))["valid"])

    def test_deepmd_finetune_package_sets_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_deepmd_committee(root, architecture="dpa2_ft", finetune=True)
            bundle = Path(
                collect_committee(source, root / "stored" / "ft_v1", engine="deepmd")["bundle"]
            )
            result = pack_huggingface(
                bundle,
                root / "hf" / "ft_v1",
                repo_id="myorg/sintin-dpa2-ft",
                base_model="myorg/dpa2-foundation",
            )
            out = Path(result["output"])
            readme = (out / "README.md").read_text(encoding="utf-8")
            self.assertIn("library_name: deepmd-kit", readme)
            self.assertIn("base_model: myorg/dpa2-foundation", readme)
            self.assertIn("fine-tuned", readme)
            self.assertIn("/foundation/dpa2_openlam.pt", readme)
            # per-member input.json is materialised for reload
            self.assertTrue((out / "models" / "model_000.input.json").is_file())

    def test_card_states_maturity_and_avoids_overclaiming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._mace_bundle(root)
            out = Path(pack_huggingface(bundle, root / "hf" / "m")["output"])
            readme = (out / "README.md").read_text(encoding="utf-8").lower()
            self.assertIn("not", readme)
            self.assertIn("validated", readme)
            self.assertNotIn("production-ready", readme)
            self.assertNotIn("production ready", readme)

    def test_embeds_supplied_metrics_in_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._mace_bundle(root)
            metrics = root / "rmse_overall.csv"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["model", "energy_rmse_mev_per_atom", "force_rmse_mev_per_angstrom"]
                )
                writer.writeheader()
                writer.writerow(
                    {"model": "m0", "energy_rmse_mev_per_atom": "2.0", "force_rmse_mev_per_angstrom": "40.0"}
                )
                writer.writerow(
                    {"model": "m1", "energy_rmse_mev_per_atom": "4.0", "force_rmse_mev_per_angstrom": "60.0"}
                )
            out = Path(
                pack_huggingface(bundle, root / "hf" / "m", metrics_path=metrics)["output"]
            )
            readme = (out / "README.md").read_text(encoding="utf-8")
            self.assertIn("model-index:", readme)
            self.assertIn("3.0", readme)  # mean energy RMSE

    def test_rejects_zip_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_committee(root)
            archive = Path(
                collect_committee(source, root / "stored" / "mace_v1")["archive"]
            )
            with self.assertRaisesRegex(ConfigurationError, "not a .zip"):
                pack_huggingface(archive, root / "hf" / "x")


# --------------------------------------------------------------------------- #
# guardrails / CLI wiring
# --------------------------------------------------------------------------- #
class PackagingGuardrailTests(unittest.TestCase):
    def test_module_makes_no_network_calls(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src" / "interfaceforge" / "packaging.py"
        text = source.read_text(encoding="utf-8")
        for forbidden in (
            "import huggingface_hub",
            "from huggingface_hub",
            "import requests",
            "urllib.request",
            "http.client",
            "import socket",
        ):
            self.assertNotIn(forbidden, text, forbidden)

    def test_cli_exposes_package_group_and_collect_archive(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["collect", "--archive", "backup.zip"])
        self.assertEqual(args.archive, "backup.zip")

        args = parser.parse_args(
            ["package", "huggingface", "bundle", "out", "--repo-id", "a/b", "--zip"]
        )
        self.assertEqual(args.package_command, "huggingface")
        self.assertTrue(args.zip)

        args = parser.parse_args(["package", "dataset-archive", "datasets/canonical", "out.zip"])
        self.assertEqual(args.package_command, "dataset-archive")

        args = parser.parse_args(["package", "verify", "some.zip"])
        self.assertEqual(args.package_command, "verify")

        args = parser.parse_args(
            ["committee", "collect", "models/deepmd/dpa2", "out", "--engine", "deepmd"]
        )
        self.assertEqual(args.engine, "deepmd")


if __name__ == "__main__":
    unittest.main()
