from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from test_committee import write_committee
from test_config_scheduler import write_campaign

from interfaceforge.cli import build_parser
from interfaceforge.committee import collect_committee, verify_committee_bundle
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.packaging import (
    materialize_dataset,
    pack_campaign,
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


def write_materializable_dataset(root: Path, *, seed: int = 0) -> tuple[Path, dict[str, dict[str, np.ndarray]]]:
    """A DeePMD-only dataset with high-precision floats and full
    move_mask.npy / system_meta.json / frame_map.csv sidecars, i.e. exactly
    what `iface collect` (post-sidecar) writes -- the dedupe/materialize
    round-trip fixture. Returns the dataset root and the exact source arrays
    per split for independent comparison against materialized extxyz."""

    dataset = root / "datasets" / "canonical"
    deepmd_root = dataset / "deepmd"
    rng = np.random.default_rng(seed)
    type_map = ["Si", "N", "Ti"]
    expected: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid", "test"):
        system = deepmd_root / split / "sys1"
        set_dir = system / "set.000"
        set_dir.mkdir(parents=True)
        natoms, nframes = 4, 2
        symbols = [type_map[i % 3] for i in range(natoms)]
        atom_types = [type_map.index(symbol) for symbol in symbols]
        (system / "type.raw").write_text("\n".join(map(str, atom_types)) + "\n", encoding="utf-8")
        (system / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")

        coord = rng.standard_normal((nframes, natoms, 3)) * np.pi
        box = np.tile((np.eye(3) * 11.987654321).reshape(1, 3, 3), (nframes, 1, 1))
        energy = rng.standard_normal((nframes, 1)) * np.e * 100
        force = rng.standard_normal((nframes, natoms, 3)) * 1e-6
        move_mask = np.ones((nframes, natoms), dtype=np.int8)
        move_mask[:, 0] = 0  # first atom frozen

        np.save(set_dir / "coord.npy", coord.reshape(nframes, -1))
        np.save(set_dir / "box.npy", box.reshape(nframes, -1))
        np.save(set_dir / "energy.npy", energy)
        np.save(set_dir / "force.npy", force.reshape(nframes, -1))
        np.save(system / "move_mask.npy", move_mask)
        (system / "system_meta.json").write_text(
            json.dumps({"kind": "bulk", "tebeg_k": 450.0, "high_temperature": True}), encoding="utf-8"
        )
        with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["local_frame", "source_frame", "source_path", "min_coordination_number", "mean_coordination_number"]
            )
            for index in range(nframes):
                writer.writerow([index, index, f"/vasp/sys1/{split}", 4, 4.5])
        expected[split] = {"coord": coord, "box": box, "energy": energy, "force": force, "move_mask": move_mask}

    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy": "grouped",
                "ratios": [0.8, 0.1, 0.1],
                "stride": 5,
                "type_map": type_map,
                "frame_counts": {"train": 2, "valid": 2, "test": 2},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "manifest.csv").write_text("run_id\nsys1\n", encoding="utf-8")
    (dataset / "frames.csv").write_text("run_id\nsys1\n", encoding="utf-8")
    return dataset, expected


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
# dedupe archive + materialize (single-copy storage, regenerated extxyz)
# --------------------------------------------------------------------------- #
class DedupeMaterializeTests(unittest.TestCase):
    def test_dedupe_excludes_extxyz_and_materialize_round_trips_exact_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, expected = write_materializable_dataset(root)

            result = pack_dataset_archive(dataset, root / "backup" / "dedupe.zip", dedupe=True)
            self.assertTrue(result["dedupe"])
            self.assertFalse(result["include_extxyz"])
            with zipfile.ZipFile(Path(result["archive"])) as handle:
                names = handle.namelist()
            self.assertFalse(any(name.endswith(".extxyz") for name in names))
            self.assertTrue(any(name.endswith("move_mask.npy") for name in names))
            self.assertTrue(verify_package(Path(result["archive"]))["valid"])

            extracted = root / "restored"
            with zipfile.ZipFile(Path(result["archive"])) as handle:
                handle.extractall(extracted)
            data_dir = extracted / "dedupe" / "data"

            payload = materialize_dataset(data_dir)
            self.assertEqual(payload["warnings"], [])
            self.assertEqual({"train", "valid", "test"}, set(payload["materialized"]))

            from ase.io import read

            for split, arrays in expected.items():
                frames = read(str(data_dir / f"{split}.extxyz"), index=":")
                self.assertEqual(len(frames), 2)
                for index, atoms in enumerate(frames):
                    self.assertTrue(np.array_equal(atoms.positions, arrays["coord"][index]))
                    self.assertTrue(np.array_equal(atoms.cell.array, arrays["box"][index]))
                    self.assertTrue(np.array_equal(atoms.arrays["REF_forces"], arrays["force"][index]))
                    self.assertEqual(atoms.info["REF_energy"], arrays["energy"][index][0])
                    reconstructed = np.ones(len(atoms), dtype=np.int8)
                    for constraint in atoms.constraints:
                        reconstructed[np.asarray(constraint.get_indices(), dtype=int)] = 0
                    self.assertTrue(np.array_equal(reconstructed, arrays["move_mask"][index]))
                    self.assertEqual(atoms.info.get("IF_kind"), "bulk")
                    self.assertTrue(atoms.info.get("IF_high_temperature"))

    def test_dedupe_rejects_dataset_missing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_canonical_dataset(root)  # no move_mask.npy / system_meta.json
            with self.assertRaisesRegex(SafetyError, "move_mask.npy and system_meta.json"):
                pack_dataset_archive(dataset, root / "x.zip", dedupe=True)

    def test_materialize_degrades_gracefully_without_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_canonical_dataset(root)
            payload = materialize_dataset(dataset, root / "materialized")
            self.assertTrue(any("move_mask.npy" in w for w in payload["warnings"]))
            self.assertTrue(any("system_meta.json" in w for w in payload["warnings"]))
            self.assertTrue((root / "materialized" / "train.extxyz").is_file())

    def test_materialize_refuses_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = write_materializable_dataset(root)
            materialize_dataset(dataset)
            with self.assertRaisesRegex(SafetyError, "Refusing to overwrite"):
                materialize_dataset(dataset)
            materialize_dataset(dataset, force=True)


def write_mapped_leaf_dataset(root: Path) -> Path:
    """The `iface-mapped-collect` / leaf-heritage layout: leaf_manifest.json
    at the MACE-side root, a second one under deepmd/, and no move_mask.npy /
    system_meta.json sidecars (those are `iface collect`-only)."""

    dataset = root / "datasets" / "canonical"
    dataset.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (dataset / f"{split}.extxyz").write_text(
            "1\nProperties=species:S:1:pos:R:3 REF_energy=-1.0\nSi 0.0 0.0 0.0\n", encoding="utf-8"
        )
    (dataset / "leaf_manifest.csv").write_text("outcar,status\n/a/OUTCAR,OK\n", encoding="utf-8")
    (dataset / "leaf_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "leaf-heritage-collector",
                "engine": "mace",
                "frame_counts": {"train": 8, "valid": 1, "test": 1},
                "ratios": [0.8, 0.1, 0.1],
                "stride": 5,
                "split_mode": "heritage",
            }
        ),
        encoding="utf-8",
    )
    deepmd = dataset / "deepmd"
    system = deepmd / "train" / "interface" / "300K" / "sys1"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n", encoding="utf-8")
    (system / "type_map.raw").write_text("Si\n", encoding="utf-8")
    for name in ("coord", "box", "energy", "force"):
        np.save(set_dir / f"{name}.npy", np.zeros((1, 3)))
    (system / "heritage.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (deepmd / "leaf_manifest.csv").write_text("outcar,status\n/a/OUTCAR,OK\n", encoding="utf-8")
    (deepmd / "leaf_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "method": "leaf-heritage-collector", "engine": "deepmd", "type_map": ["Si"]}
        ),
        encoding="utf-8",
    )
    return dataset


class MappedLeafDatasetArchiveTests(unittest.TestCase):
    def test_mirrors_leaf_heritage_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_mapped_leaf_dataset(root)
            result = pack_dataset_archive(dataset, root / "backup" / "periodic.zip")
            self.assertEqual(result["dedupe"], False)
            self.assertTrue(verify_package(Path(result["archive"]))["valid"])
            with zipfile.ZipFile(Path(result["archive"])) as handle:
                names = handle.namelist()
            self.assertIn("periodic/data/leaf_manifest.json", names)
            self.assertIn("periodic/data/deepmd/leaf_manifest.json", names)
            self.assertIn(
                "periodic/data/deepmd/train/interface/300K/sys1/set.000/coord.npy", names
            )
            self.assertIn("periodic/data/train.extxyz", names)

    def test_dedupe_refuses_leaf_heritage_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_mapped_leaf_dataset(root)
            with self.assertRaisesRegex(ConfigurationError, "written by 'iface collect'"):
                pack_dataset_archive(dataset, root / "x.zip", dedupe=True)


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

        args = parser.parse_args(
            ["package", "dataset-archive", "datasets/canonical", "out.zip", "--dedupe"]
        )
        self.assertEqual(args.package_command, "dataset-archive")
        self.assertTrue(args.dedupe)

        args = parser.parse_args(["package", "materialize", "datasets/canonical"])
        self.assertEqual(args.package_command, "materialize")
        self.assertIsNone(args.output)

        args = parser.parse_args(["package", "verify", "some.zip"])
        self.assertEqual(args.package_command, "verify")

        args = parser.parse_args(
            ["committee", "collect", "models/deepmd/dpa2", "out", "--engine", "deepmd"]
        )
        self.assertEqual(args.engine, "deepmd")

        args = parser.parse_args(
            [
                "package", "campaign", "--repo-prefix", "myorg/sintin", "--tag", "v1",
                "--dedupe", "--deepmd-root", "models/deepmd_custom",
            ]
        )
        self.assertEqual(args.package_command, "campaign")
        self.assertEqual(args.repo_prefix, "myorg/sintin")
        self.assertTrue(args.dedupe)
        self.assertEqual(args.deepmd_root, "models/deepmd_custom")


# --------------------------------------------------------------------------- #
# iface package campaign: everything a campaign has, in one call
# --------------------------------------------------------------------------- #
def _write_full_campaign_tree(root: Path) -> Path:
    """A campaign.yaml plus the mace_committee_520eV/{base,ft} and
    models/deepmd/<arch> trees `iface mlip-progress`/`iface train` already use,
    and a plain 'iface collect'-flavored canonical dataset."""

    campaign_path = write_campaign(
        root,
        mace={"enabled": True},
        deepmd={"enabled": True, "backend": "pytorch", "architectures": ["dpa2"]},
    )

    mace_root = root / "models" / "mace_committee_520eV"
    for base in ("mace_committee", "mace_finetune_committee"):
        for seed in (0, 211, 307, 419):
            model_dir = mace_root / base / f"seed_{seed}" / "mace_model"
            model_dir.mkdir(parents=True)
            (model_dir / "sintin_mace_stagetwo.model").write_bytes(f"model-{base}-{seed}".encode())

    dpa2 = root / "models" / "deepmd" / "dpa2"
    for index in range(4):
        model_dir = dpa2 / f"model_{index:03d}"
        model_dir.mkdir(parents=True)
        (model_dir / "frozen_model.pth").write_bytes(f"frozen-{index}".encode())
        (model_dir / "input.json").write_text(
            json.dumps({"model": {"type_map": ["Si", "N"]}, "training": {"numb_steps": 100}}),
            encoding="utf-8",
        )
    (root / "models" / "deepmd" / "ensemble_manifest.json").write_text(
        json.dumps(
            {
                "engine": "deepmd",
                "backend": "pytorch",
                "architectures": ["dpa2"],
                "models": [{"architecture": "dpa2", "index": i, "seed": 11 + 12 * i} for i in range(4)],
            }
        ),
        encoding="utf-8",
    )

    write_canonical_dataset(root)
    return campaign_path


class PackageCampaignTests(unittest.TestCase):
    def test_collects_and_packages_every_committee_plus_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(_write_full_campaign_tree(root))

            payload = pack_campaign(campaign, repo_prefix="myorg/sintin", tag="v1")

            self.assertEqual(payload["errors"], [])
            self.assertEqual(payload["skipped"], [])
            self.assertIsNotNone(payload["dataset_archive"])
            components = {entry["component"] for entry in payload["committees"]}
            self.assertEqual(components, {"mace", "mace-ft", "dpa2"})
            for entry in payload["committees"]:
                self.assertTrue(verify_committee_bundle(Path(entry["bundle"]["bundle"]))["valid"])
                self.assertEqual(entry["huggingface"]["repo_id"], f"myorg/sintin-{entry['component']}")
                self.assertTrue(Path(entry["huggingface"]["output"], "README.md").is_file())
            self.assertTrue(verify_package(Path(payload["dataset_archive"]["archive"]))["valid"])

    def test_missing_pieces_are_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_campaign(root, mace={"enabled": True}))
            # No mace_committee_520eV/, no datasets/canonical/, deepmd disabled.
            payload = pack_campaign(campaign)
            self.assertEqual(payload["committees"], [])
            self.assertEqual(payload["errors"], [])
            steps = {row["step"] for row in payload["skipped"]}
            self.assertIn("dataset_archive", steps)
            self.assertIn("mace_committee", steps)
            self.assertIn("mace_finetune_committee", steps)

    def test_deepmd_committee_is_found_by_directory_not_by_campaign_config(self) -> None:
        """A trained DeePMD committee on disk is packaged even when
        models.deepmd isn't enabled (or configured at all) in campaign.yaml --
        the same directory-only discovery already used for MACE, not gated on
        campaign.yaml staying in sync with what was actually trained."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # mace enabled, deepmd left entirely unconfigured/disabled.
            campaign = load_campaign(write_campaign(root, mace={"enabled": True}))

            dpa2 = root / "models" / "deepmd" / "dpa2"
            for index in range(4):
                model_dir = dpa2 / f"model_{index:03d}"
                model_dir.mkdir(parents=True)
                (model_dir / "frozen_model.pth").write_bytes(f"frozen-{index}".encode())

            payload = pack_campaign(campaign, include_huggingface=False, include_dataset_archive=False)

            self.assertEqual(payload["errors"], [])
            components = {entry["component"] for entry in payload["committees"]}
            self.assertIn("dpa2", components)
            self.assertNotIn("deepmd", {row["step"] for row in payload["skipped"]})

    def test_deepmd_root_override_and_ignores_evaluation_and_smoke_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(write_campaign(root, mace={"enabled": True}))

            custom_root = root / "elsewhere" / "deepmd_models"
            dpa3 = custom_root / "dpa3"
            for index in range(4):
                model_dir = dpa3 / f"model_{index:03d}"
                model_dir.mkdir(parents=True)
                (model_dir / "frozen_model.pth").write_bytes(f"frozen-{index}".encode())
            # These must not be mistaken for architecture directories.
            (custom_root / "evaluation" / "dpa3" / "job_1").mkdir(parents=True)
            (custom_root / "smoke" / "job_1" / "dpa3").mkdir(parents=True)

            payload = pack_campaign(
                campaign,
                deepmd_root=custom_root,
                include_huggingface=False,
                include_dataset_archive=False,
            )

            self.assertEqual(payload["errors"], [])
            components = {entry["component"] for entry in payload["committees"]}
            self.assertEqual(components, {"dpa3"})

    def test_rerun_with_same_tag_reports_immutability_errors_not_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = load_campaign(_write_full_campaign_tree(root))
            first = pack_campaign(campaign, tag="v1", include_huggingface=False)
            self.assertEqual(first["errors"], [])

            second = pack_campaign(campaign, tag="v1", include_huggingface=False)
            self.assertEqual(len(second["errors"]), 4)  # dataset + 2 mace + 1 deepmd
            for row in second["errors"]:
                self.assertRegex(row["detail"], "already exists")
            # the original bundles must be untouched
            self.assertTrue(verify_package(Path(first["dataset_archive"]["archive"]))["valid"])

    def test_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_full_campaign_tree(root)
            parser = build_parser()
            args = parser.parse_args(["package", "campaign", "-c", str(root / "campaign.yaml")])
            from interfaceforge.cli import cmd_package

            exit_code = cmd_package(args)
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "packaged" / "backups").is_dir())
            self.assertTrue((root / "packaged" / "hf").is_dir())


if __name__ == "__main__":
    unittest.main()
