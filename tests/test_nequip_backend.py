# ruff: noqa: E501
"""NequIP committee backend: config, dataset handoff, committee, Slurm, resume, discovery, evaluation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from nio_fixture import build_nio_tree

from interfaceforge.cli import main as cli_main
from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.nequip import (
    DEFAULTS,
    NEQUIP_EVALUATOR,
    discover_members,
    evaluate_nequip_committee,
    generate_nequip_training,
    member_state,
    parse_lightning_metrics,
    render_status,
    resolve_settings,
    submit_nequip,
)
from interfaceforge.nio_dataset import ExportConfig, export_dataset

REPO = Path(__file__).resolve().parents[1]
FAKE_TRAIN = """#!/usr/bin/env bash
echo "$@" >> "$FAKE_LOG"
out=""; for a in "$@"; do case "$a" in hydra.run.dir=*) out="${a#hydra.run.dir=}";; esac; done
mkdir -p "$out" "$(dirname "$out")/logs/csv/version_0"
echo ckpt > "$out/last.ckpt"
printf 'epoch,step,val0_epoch/forces_mae,val0_epoch/weighted_sum\\n3,40,0.08,0.1\\n' > "$(dirname "$out")/logs/csv/version_0/metrics.csv"
if [[ "${FAKE_FAIL:-0}" == "1" ]]; then echo "fake OOM" >&2; exit 1; fi
echo best > "$out/best.ckpt"
"""
FAKE_PACKAGE = '#!/usr/bin/env bash\necho "package $@" >> "$FAKE_LOG"; echo "pkg $2" > "$3"\n'
FAKE_COMPILE = '#!/usr/bin/env bash\necho "compile $@" >> "$FAKE_LOG"; echo "compiled $1" > "$2"\n'
STUB_CALCULATOR = """
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class NequIPCalculator(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    @classmethod
    def from_compiled_model(cls, compile_path, device="cpu", **kwargs):
        text = open(compile_path).read()
        return cls(offset=0.001 * (1 + sum(map(ord, text)) % 5))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(atoms.info["REF_energy"]) + self.offset * len(atoms),
            "free_energy": float(atoms.info["REF_energy"]),
            "forces": np.asarray(atoms.arrays["REF_forces"]) + 10 * self.offset,
        }
"""


def make_campaign(root: Path, *, nequip: dict | None = None, profile: str = "loni.yaml") -> Path:
    head = build_nio_tree(
        root / "NiO_head",
        cases=[
            ("OH0", "NiO_m110_Big_U46", ("Ni", "O"), (4, 4)),
            ("OH50", "NiO_m110_Big_U46_OH50_clustered_capped", ("H", "Ni", "O"), (2, 4, 6)),
            ("OH50", "NiO_m110_Big_U46_OH50_clustered_dissoc", ("H", "Ni", "O"), (4, 4, 6)),
            (
                "OH25",
                "NiO_m110_Big_U46_OH25_scattered_dissoc_DCZ4P_bare",
                ("C", "H", "N", "Ni", "O", "P"),
                (2, 3, 1, 4, 7, 1),
            ),
        ],
        temperatures=(300, 450),
    )
    campaign = root / "campaign"
    (campaign / "profiles").mkdir(parents=True)
    shutil.copy(REPO / "profiles" / profile, campaign / "profiles" / profile)
    (campaign / "structures").mkdir()
    (campaign / "structures" / "nio.vasp").write_text("placeholder\n")
    export_dataset([head], campaign / "datasets" / "canonical", ExportConfig())
    settings = {"enabled": True, "dataset": "datasets/canonical", "r_max": 5.0, "max_epochs": 300}
    settings.update(nequip or {})
    (campaign / "campaign.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "project": {"name": "nio-test"},
                "profile": f"profiles/{profile}",
                "systems": [{"id": "nio", "kind": "surface", "structure": "structures/nio.vasp"}],
                "models": {"nequip": settings},
            }
        )
    )
    return campaign


def install_fakes(root: Path) -> dict[str, str]:
    bin_dir = root / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in {
        "nequip-train": FAKE_TRAIN,
        "nequip-package": FAKE_PACKAGE,
        "nequip-compile": FAKE_COMPILE,
    }.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    stub = root / "stubpy" / "nequip" / "integrations"
    stub.mkdir(parents=True, exist_ok=True)
    (root / "stubpy" / "nequip" / "__init__.py").write_text("__version__ = '0.0-stub'\n")
    (stub / "__init__.py").write_text("")
    (stub / "ase.py").write_text(STUB_CALCULATOR)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "FAKE_LOG": str(root / "fake.log"),
            "PYTHONPATH": f"{root / 'stubpy'}:{env.get('PYTHONPATH', '')}",
        }
    )
    return env


class SettingsTests(unittest.TestCase):
    def test_r_max_is_required_and_defaults_are_listed(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "r_max"):
            resolve_settings({"enabled": True})
        resolved, defaults = resolve_settings({"enabled": True, "r_max": 4.5, "num_layers": 5})
        self.assertEqual(resolved["r_max"], 4.5)
        self.assertEqual(resolved["num_layers"], 5)
        self.assertNotIn("num_layers", defaults)
        self.assertIn("learning_rate", defaults)
        self.assertEqual(set(defaults) | {"num_layers"}, set(DEFAULTS))

    def test_invalid_settings_are_rejected(self) -> None:
        for bad in (
            {"r_max": 5, "bogus": 1},
            {"r_max": -1},
            {"r_max": 5, "device": "tpu"},
            {"r_max": 5, "model_dtype": "float16"},
            {"r_max": 5, "compile_mode": "jit"},
            {"r_max": 5, "tf32": True, "model_dtype": "float64"},
            {"r_max": 5, "extra_model": {"r_max": 9}},
        ):
            with self.assertRaises(ConfigurationError, msg=str(bad)):
                resolve_settings(bad)

    def test_campaign_validation_of_seeds_and_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = make_campaign(Path(temporary))
            data = yaml.safe_load((campaign / "campaign.yaml").read_text())
            for patch, message in (
                ({"seeds": [11, 11]}, "unique"),
                ({"seeds": [11], "committee": 2}, "cover"),
                ({"device": "gpu"}, "cuda or cpu"),
                ({"r_max": None}, "r_max"),
            ):
                variant = json.loads(json.dumps(data))
                variant["models"]["nequip"].update(patch)
                path = campaign / "variant.yaml"
                path.write_text(yaml.safe_dump(variant))
                with self.assertRaisesRegex(ConfigurationError, message):
                    load_campaign(path)


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.campaign_dir = make_campaign(self.root)
        self.campaign = load_campaign(self.campaign_dir / "campaign.yaml")

    def test_config_is_nequip_not_allegro_and_uses_canonical_splits(self) -> None:
        manifest = generate_nequip_training(self.campaign)
        self.assertEqual(manifest["architecture"], "nequip")
        self.assertEqual([member["seed"] for member in manifest["members"]], [11, 23, 37, 53])
        config = yaml.safe_load(Path(manifest["members"][1]["config"]).read_text())
        model = config["training_module"]["model"]
        self.assertEqual(model["_target_"], "nequip.model.NequIPGNNModel")
        self.assertNotIn("allegro", json.dumps(config).lower())
        self.assertEqual(model["seed"], 23)
        self.assertEqual(config["data"]["seed"], 23)
        self.assertEqual(config["cutoff_radius"], 5.0)
        self.assertEqual(config["model_type_names"], ["C", "H", "N", "Ni", "O", "P"])
        canonical = self.campaign_dir / "datasets" / "canonical"
        self.assertEqual(config["data"]["train_file_path"], str(canonical / "train.extxyz"))
        self.assertEqual(config["data"]["val_file_path"], str(canonical / "valid.extxyz"))
        self.assertEqual(config["data"]["test_file_path"], str(canonical / "test.extxyz"))
        self.assertEqual(config["data"]["key_mapping"], {"REF_energy": "total_energy", "REF_forces": "forces"})
        self.assertEqual(config["run"], ["train", "test"])
        self.assertEqual(config["trainer"]["accelerator"], "gpu")
        self.assertEqual(config["trainer"]["max_epochs"], 300)
        callbacks = {callback["_target_"]: callback for callback in config["trainer"]["callbacks"]}
        checkpoint = callbacks["lightning.pytorch.callbacks.ModelCheckpoint"]
        self.assertEqual((checkpoint["filename"], checkpoint["save_last"]), ("best", True))
        self.assertEqual(config["trainer"]["logger"]["_target_"], "lightning.pytorch.loggers.CSVLogger")

    def test_dataset_and_split_identity_are_recorded(self) -> None:
        manifest = generate_nequip_training(self.campaign)
        canonical = json.loads((self.campaign_dir / "datasets" / "canonical" / "manifest.json").read_text())
        identity = manifest["dataset"]["identity"]
        self.assertEqual(identity["dataset_hash"], canonical["dataset_hash"])
        self.assertEqual(identity["split_hash"], canonical["split_hash"])
        self.assertEqual(manifest["dataset"]["source"], "canonical-manifest")
        self.assertIn("commit", manifest["interfaceforge_commit"])
        self.assertIn("r_max", manifest["hyperparameters"])
        self.assertIn("learning_rate", manifest["defaults_applied"])
        self.assertEqual(manifest["verification_status"].split(":")[0], "automated-test-only")

    def test_modified_split_file_is_refused(self) -> None:
        train = self.campaign_dir / "datasets" / "canonical" / "train.extxyz"
        train.write_text(train.read_text() + "\n")
        with self.assertRaisesRegex(SafetyError, "no longer matches"):
            generate_nequip_training(self.campaign)

    def test_type_names_must_match_the_dataset(self) -> None:
        data = yaml.safe_load((self.campaign_dir / "campaign.yaml").read_text())
        data["models"]["nequip"]["type_names"] = ["Ni", "O"]
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(SafetyError, "disagree"):
            generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"))

    def test_configurable_committee_and_deterministic_paths(self) -> None:
        data = yaml.safe_load((self.campaign_dir / "campaign.yaml").read_text())
        data["models"]["nequip"].update({"seeds": [5, 7, 9], "committee": 2, "data_seed": 1})
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        campaign = load_campaign(self.campaign_dir / "campaign.yaml")
        first = generate_nequip_training(campaign)
        self.assertEqual([member["seed"] for member in first["members"]], [5, 7])
        self.assertEqual([Path(member["directory"]).name for member in first["members"]], ["seed_5", "seed_7"])
        self.assertEqual({member["data_seed"] for member in first["members"]}, {1})
        with self.assertRaisesRegex(SafetyError, "not empty"):
            generate_nequip_training(campaign)
        second = generate_nequip_training(campaign, force=True)
        self.assertEqual(
            [member["config_sha256"] for member in first["members"]],
            [member["config_sha256"] for member in second["members"]],
        )
        self.assertIn("--array=0-1%2", Path(second["launchers"]["committee"]).read_text())

    def test_slurm_scripts_come_from_the_profile(self) -> None:
        manifest = generate_nequip_training(self.campaign)
        texts = {name: Path(path).read_text() for name, path in manifest["launchers"].items()}
        committee = texts["committee"]
        for expected in (
            "#SBATCH --partition=gpu2",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --array=0-3%2",
            "NEQUIP_ACTIVATE_SCRIPT",
            "SEEDS=(11 23 37 53)",
            'train_member.sh" all',
        ):
            self.assertIn(expected, committee)
        self.assertIn("#SBATCH --array", texts["evaluate"])
        self.assertNotIn("#SBATCH --array", texts["preflight"])
        self.assertIn("sha256sum -c", texts["preflight"])
        self.assertIn("--max-frames 20", texts["smoke"])
        self.assertIn("hydra.run.dir=$SMOKE_DIR/outputs", texts["smoke"])
        for text in texts.values():
            self.assertNotIn("lgutsev", text)
            self.assertNotIn("loni_perovsk27", text)
            self.assertNotIn("sbatch ", text.split("\n", 1)[1].replace("Submit with sbatch", ""))
        driver = (self.campaign_dir / "models" / "nequip" / "seed_11" / "train_member.sh").read_text()
        self.assertIn("++ckpt_path=$LAST", driver)
        self.assertIn("--mode aotinductor --target ase", driver)
        subprocess.run(
            ["bash", "-n", str(self.campaign_dir / "models" / "nequip" / "seed_11" / "train_member.sh")], check=True
        )
        for path in manifest["launchers"].values():
            subprocess.run(["bash", "-n", path], check=True)
        compile(NEQUIP_EVALUATOR, "evaluate_nequip.py", "exec")

    def test_device_must_match_profile_resources(self) -> None:
        data = yaml.safe_load((self.campaign_dir / "campaign.yaml").read_text())
        data["models"]["nequip"].update({"device": "cpu"})
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        # device=cpu without a profile selects nequip_cpu automatically ...
        default_cpu = generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"))
        self.assertEqual(default_cpu["runtime"]["profile"], "nequip_cpu")
        # ... but an explicit GPU profile with device=cpu is a contradiction.
        data["models"]["nequip"].update({"device": "cpu", "profile": "nequip_gpu"})
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(SafetyError, "requests 1 GPU"):
            generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"), force=True)
        data["models"]["nequip"].update({"device": "cuda", "profile": "nequip_cpu"})
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(SafetyError, "requests no GPU"):
            generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"), force=True)
        data["models"]["nequip"].update({"device": "cpu", "profile": "nequip_cpu", "model_dtype": "float64"})
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        manifest = generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"), force=True)
        config = yaml.safe_load(Path(manifest["members"][0]["config"]).read_text())
        self.assertEqual(config["trainer"]["accelerator"], "cpu")
        self.assertEqual(config["training_module"]["model"]["model_dtype"], "float64")
        self.assertNotIn("--gres", Path(manifest["launchers"]["committee"]).read_text())

    def test_local_profile_runs_members_sequentially(self) -> None:
        campaign_dir = make_campaign(self.root / "local", profile="local.yaml")
        manifest = generate_nequip_training(load_campaign(campaign_dir / "campaign.yaml"))
        text = Path(manifest["launchers"]["committee"]).read_text()
        self.assertNotIn("#SBATCH", text)
        self.assertIn('for TASK_ID in "${!SEEDS[@]}"; do', text)
        subprocess.run(["bash", "-n", manifest["launchers"]["committee"]], check=True)


class DriverAndDiscoveryTests(unittest.TestCase):
    """Runs the generated bash driver with fake nequip-* executables (no NequIP needed)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.campaign_dir = make_campaign(self.root, nequip={"device": "cpu", "profile": "nequip_cpu"})
        self.manifest = generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"))
        self.nequip_root = Path(self.manifest["root"])
        self.env = install_fakes(self.root)

    def _run(self, seed: int, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.nequip_root / f"seed_{seed}" / "train_member.sh"), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def test_failure_resume_completion_and_config_guard(self) -> None:
        failed = self._run(11, FAKE_FAIL="1")
        self.assertEqual(failed.returncode, 1)
        status = json.loads((self.nequip_root / "seed_11" / "status.json").read_text())
        self.assertEqual((status["state"], status["stage"]), ("failed", "train"))
        self.assertEqual(member_state(self.nequip_root / "seed_11")["state"], "failed")

        resumed = self._run(11)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        log = (self.root / "fake.log").read_text().splitlines()
        self.assertNotIn("++ckpt_path", log[0])
        self.assertIn(f"++ckpt_path={self.nequip_root / 'seed_11' / 'outputs' / 'last.ckpt'}", log[1])
        state = member_state(self.nequip_root / "seed_11")
        self.assertEqual(state["state"], "complete")
        self.assertEqual(state["restarts"], 1)
        self.assertTrue(state["package"] and state["compiled_model"].endswith("model.nequip.pt2"))
        self.assertEqual(state["epoch"], 3)
        self.assertAlmostEqual(state["val_metrics"]["forces_mae"], 0.08)
        self.assertTrue((self.nequip_root / "seed_11" / "versions.json").is_file())

        again = self._run(11)
        self.assertEqual(again.returncode, 0)
        self.assertIn("already complete", again.stderr)
        self.assertEqual(len((self.root / "fake.log").read_text().splitlines()), 4)

        config = self.nequip_root / "seed_23" / "config.yaml"
        config.write_text(config.read_text() + "# edited\n")
        refused = self._run(23)
        self.assertEqual(refused.returncode, 3)
        self.assertEqual(
            json.loads((self.nequip_root / "seed_23" / "status.json").read_text())["stage"], "config_check"
        )

    def test_regeneration_refuses_changed_config_under_existing_checkpoints(self) -> None:
        self._run(11, FAKE_FAIL="1")
        data = yaml.safe_load((self.campaign_dir / "campaign.yaml").read_text())
        data["models"]["nequip"]["learning_rate"] = 0.001
        (self.campaign_dir / "campaign.yaml").write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(SafetyError, "different config"):
            generate_nequip_training(load_campaign(self.campaign_dir / "campaign.yaml"), force=True)

    def test_discovery_reports_every_member_state(self) -> None:
        self._run(11)
        running = self.nequip_root / "seed_23"
        (running / "status.json").write_text(json.dumps({"state": "running", "stage": "train", "seed": 23}))
        stale = self.nequip_root / "seed_37"
        (stale / "status.json").write_text(json.dumps({"state": "running", "stage": "train", "seed": 37}))
        old = 1_000_000_000
        os.utime(stale / "status.json", (old, old))
        (self.nequip_root / "seed_53" / "outputs").mkdir()
        (self.nequip_root / "seed_53" / "outputs" / "last.ckpt").write_text("x")
        state = discover_members(self.nequip_root)
        states = {member["seed"]: member["state"] for member in state["members"]}
        self.assertEqual(states, {11: "complete", 23: "running", 37: "stalled?", 53: "incomplete"})
        self.assertFalse(state["complete"])
        shutil.rmtree(self.nequip_root / "seed_53")
        self.assertEqual(discover_members(self.nequip_root)["missing_seeds"], [53])
        text = render_status(discover_members(self.nequip_root))
        self.assertIn("seed 11", text)
        self.assertIn("complete", text)

    def test_lightning_metrics_parser_handles_restarts(self) -> None:
        logs = self.root / "logs"
        for version, rows in (
            (0, "epoch,step,val0_epoch/forces_mae\n0,10,0.5\n1,20,0.3\n"),
            (1, "epoch,step,val0_epoch/forces_mae,test0_epoch/forces_rmse\n2,30,0.2,\n3,40,0.15,0.21\n"),
        ):
            path = logs / "csv" / f"version_{version}" / "metrics.csv"
            path.parent.mkdir(parents=True)
            path.write_text(rows)
        parsed = parse_lightning_metrics(logs)
        self.assertEqual((parsed["epoch"], parsed["step"], parsed["files"]), (3, 40, 2))
        self.assertAlmostEqual(parsed["val"]["forces_mae"], 0.15)
        self.assertAlmostEqual(parsed["test"]["forces_rmse"], 0.21)
        self.assertEqual(parse_lightning_metrics(self.root / "missing")["epoch"], None)

    def test_generated_evaluator_and_committee_summary(self) -> None:
        for seed in (11, 23, 37, 53):
            self.assertEqual(self._run(seed).returncode, 0)
        test_file = self.manifest["evaluation"]["test_file"]
        for index, member in enumerate(self.manifest["members"]):
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.nequip_root / "evaluate_nequip.py"),
                    "--model",
                    str(Path(member["directory"]) / "final" / "model.nequip.pt2"),
                    "--frames",
                    test_file,
                    "--expected-sha256",
                    self.manifest["evaluation"]["test_sha256"],
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
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads(
            (Path(self.manifest["members"][0]["directory"]) / "evaluation" / "predictions.json").read_text()
        )
        self.assertFalse(meta["constraints_applied_to_predictions"])
        summary = evaluate_nequip_committee(self.nequip_root)
        self.assertEqual(summary["backend"], "NequIP")
        self.assertEqual(len(summary["per_model"]), 5)
        self.assertFalse(summary["disagreement"]["calibrated"])
        self.assertIn("NOT a calibrated uncertainty", summary["notes"]["spread"])
        self.assertEqual(summary["dataset_identity"]["split_hash"], self.manifest["dataset"]["identity"]["split_hash"])
        with np.load(self.nequip_root / "evaluation" / "predictions_aligned.npz") as data:
            self.assertEqual(data["member_energy"].shape[0], 4)
        self.assertTrue(discover_members(self.nequip_root)["evaluated"])

    def test_evaluation_refuses_missing_members_and_changed_test_file(self) -> None:
        with self.assertRaisesRegex(SafetyError, "incomplete"):
            evaluate_nequip_committee(self.nequip_root)
        wrong = subprocess.run(
            [
                sys.executable,
                str(self.nequip_root / "evaluate_nequip.py"),
                "--model",
                str(self.root / "missing.pt2"),
                "--frames",
                self.manifest["evaluation"]["test_file"],
                "--output",
                str(self.root / "x"),
            ],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(wrong.returncode, 0)

    def test_submission_is_dry_run_unless_executed(self) -> None:
        fake_sbatch = self.root / "fakebin" / "sbatch"
        fake_sbatch.write_text('#!/usr/bin/env bash\necho "Submitted batch job 4242"\n')
        fake_sbatch.chmod(0o755)
        dry = submit_nequip(self.nequip_root, "committee")
        self.assertFalse(dry["executed"])
        self.assertFalse((self.nequip_root / "submissions.jsonl").exists())
        with self.assertRaisesRegex(SafetyError, "no compiled model"):
            submit_nequip(self.nequip_root, "evaluate")
        done = submit_nequip(self.nequip_root, "smoke", execute=True, sbatch=str(fake_sbatch))
        self.assertTrue(done["executed"])
        self.assertIn("4242", done["stdout"])
        record = json.loads((self.nequip_root / "submissions.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["stage"], "smoke")
        config = self.nequip_root / "seed_37" / "config.yaml"
        config.write_text(config.read_text() + "# drift\n")
        with self.assertRaisesRegex(SafetyError, "differs from the generated config"):
            submit_nequip(self.nequip_root, "committee")
        with self.assertRaises(ConfigurationError):
            submit_nequip(self.nequip_root, "everything")


class CliTests(unittest.TestCase):
    def test_help_and_argument_validation(self) -> None:
        for argv in (
            ["train", "--help"],
            ["nequip", "--help"],
            ["nequip", "submit", "--help"],
            ["dataset", "export", "--help"],
        ):
            buffer = StringIO()
            with redirect_stdout(buffer), self.assertRaises(SystemExit) as exit_code:
                cli_main(argv)
            self.assertEqual(exit_code.exception.code, 0)
        self.assertIn(
            "nequip",
            subprocess.run(
                [sys.executable, "-m", "interfaceforge", "train", "--help"], capture_output=True, text=True
            ).stdout,
        )
        for argv in (["train", "allegro"], ["nequip", "submit", "bogus"], ["dataset", "export", "x"]):
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as exit_code:
                cli_main(argv)
            self.assertEqual(exit_code.exception.code, 2)

    def test_train_and_status_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = make_campaign(Path(temporary))
            with redirect_stdout(StringIO()):
                self.assertEqual(cli_main(["train", "nequip", "-c", str(campaign / "campaign.yaml")]), 0)
            buffer = StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(cli_main(["nequip", "status", "-c", str(campaign / "campaign.yaml"), "--json"]), 0)
            payload = json.loads(buffer.getvalue())
            self.assertEqual({member["state"] for member in payload["members"]}, {"not-started"})
            buffer = StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(cli_main(["nequip", "submit", "preflight", "--root", payload["root"]]), 0)
            self.assertFalse(json.loads(buffer.getvalue())["executed"])


if __name__ == "__main__":
    unittest.main()
