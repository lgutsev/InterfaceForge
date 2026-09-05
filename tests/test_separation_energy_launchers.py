from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "launch_scripts"
SEEDS = (11, 23, 37, 53)


class SeparationEnergyLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camp = self.root / "campaign with spaces,commas"
        self.mace = self.camp / "models/mace_committee_520eV/mace_committee"
        self.put(self.camp / "campaign.yaml")
        for term in ("N_term_dft", "Ti_term_dft"):
            self.put(self.camp / "adhesion" / term / "manifest.json")
        for seed in SEEDS:
            self.put(self.mace / f"seed_{seed}/mace_model/final_stagetwo.model")
        for i in range(4):
            self.put(self.camp / f"models/deepmd/dpa2/model_{i:03d}/model.ckpt.pt")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("SEPARATION_", "MACE_COMMITTEE_"))}
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}", CALLS=str(self.calls))
        self.command("sbatch", '''#!/usr/bin/env python3
import json, os, sys
label = os.path.basename(sys.argv[-1]).split('.')[0].removeprefix('separation_energy_')
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'run': os.environ['SEPARATION_RUN_DIR'],
                        'root': os.environ['MACE_COMMITTEE_ROOT']}) + '\\n')
if os.environ.get('FAIL_JOB') == label:
    print('scheduler rejected request'); sys.exit(1)
id = {'mace': '1001', 'deepmd': '1002', 'merge': '1003'}[label]
mode = os.environ.get('SBATCH_MODE', 'loni')
if mode == 'malformed':
    print('accepted maybe'); sys.exit(0)
if mode == 'conflict':
    print('Submitted batch job 9999')
print('sbatch: 4406304.39 SUs available in loni_perovsk27')
print('sbatch: 128.00 SUs estimated for this job.')
if mode == 'standard':
    print('Submitted batch job ' + id)
else:
    print('sbatch: lua: Submitted job ' + id)
    print(id + ';cluster' if mode == 'cluster' else id)
''')

    @staticmethod
    def put(path: Path, data: bytes = b"model") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def command(self, name: str, script: str) -> None:
        path = self.bin / name
        path.write_text(script)
        path.chmod(0o755)

    def submit(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(LAUNCHERS / "submit_separation_energy.sh"), *args],
            cwd=self.camp, env={**self.env, **env}, text=True, capture_output=True,
        )

    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_bash_syntax(self) -> None:
        for name in ("submit_separation_energy.sh", "separation_energy_common.sh",
                     "separation_energy_mace.sbatch", "separation_energy_deepmd.sbatch",
                     "separation_energy_merge.sbatch", "freeze_missing_deepmd_dpa2.sbatch"):
            subprocess.run(["bash", "-n", str(LAUNCHERS / name)], check=True)

    def test_dry_run_checks_real_layout_without_submission_or_output(self) -> None:
        result = self.submit("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.mace), result.stdout)
        self.assertIn("model.ckpt.pt", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.camp / "audit").exists())

    def test_legacy_root_and_both_override_conventions(self) -> None:
        for root in (self.mace, self.mace.parent):
            result = self.submit("--dry-run", MACE_COMMITTEE_ROOT=str(root.relative_to(self.camp)))
            self.assertEqual(result.returncode, 0, result.stderr)
        legacy = self.camp / "models/mace_committee"
        self.mace.rename(legacy)
        result = self.submit("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(legacy), result.stdout)

    def test_empty_legacy_root_ignored_but_two_complete_roots_need_selection(self) -> None:
        legacy = self.camp / "models/mace_committee"
        legacy.mkdir()
        self.assertEqual(self.submit("--dry-run").returncode, 0)
        for seed in SEEDS:
            self.put(legacy / f"seed_{seed}/other.model")
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("2 complete MACE", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_export_priority_excludes_backups_compiled_and_checkpoints(self) -> None:
        seed = self.mace / "seed_11"
        for name in ("mace_model/base.model", "mace_model/x_stagetwo_compiled.model",
                     "checkpoints/copy_stagetwo.model", "mace_model/backup/old_stagetwo.model"):
            self.put(seed / name)
        result = self.submit("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(seed / "mace_model/final_stagetwo.model"), result.stdout)
        self.assertNotIn("copy_stagetwo", result.stdout)
        self.assertNotIn("has 2", result.stderr)

    def test_newest_stage_export_and_symlink_supported(self) -> None:
        directory = self.mace / "seed_11/mace_model"
        old = directory / "old_stagetwo.model"
        self.put(old)
        os.utime(old, (1, 1))
        model = directory / "final_stagetwo.model"
        target = self.root / "actual.model"
        model.rename(target)
        model.symlink_to(target)
        result = self.submit("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("using the newest", result.stderr)
        self.assertIn(str(model), result.stdout)

    def test_empty_export_dangling_link_and_checkpoint_only_fail_before_sbatch(self) -> None:
        model = self.mace / "seed_11/mace_model/final_stagetwo.model"
        model.write_bytes(b"")
        (model.parent / "broken.model").symlink_to(self.root / "absent")
        self.put(self.mace / "seed_11/checkpoints/copy_stagetwo.model")
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("seed 11", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_missing_deepmd_fails_before_any_submission(self) -> None:
        (self.camp / "models/deepmd/dpa2/model_003/model.ckpt.pt").unlink()
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_003", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_submission_pins_models_exports_paths_and_isolates_retries(self) -> None:
        self.put(self.camp / "models/deepmd/dpa2/model_000/frozen_model.pth")
        for _ in range(2):
            result = self.submit()
            self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual(len(records), 6)
        self.assertNotEqual(records[0]["run"], records[3]["run"])
        self.assertIn("--dependency=afterok:1001:1002", records[2]["args"])
        self.assertIn("--export=ALL", records[0]["args"])
        run = Path(records[0]["run"])
        self.assertEqual(len((run / "mace_models.txt").read_text().splitlines()), 4)
        deepmd = (run / "deepmd_models.txt").read_text().splitlines()
        self.assertTrue(deepmd[0].endswith("frozen_model.pth"))
        self.assertTrue(deepmd[1].endswith("model.ckpt.pt"))
        self.assertEqual((run / "jobs.tsv").read_text(), "MACE\t1001\nDeePMD\t1002\nmerge\t1003\n")

    def test_parser_accepts_cluster_and_standard_output(self) -> None:
        for mode in ("cluster", "standard"):
            result = self.submit(SBATCH_MODE=mode)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_malformed_and_conflicting_job_ids_stop_and_keep_raw_log(self) -> None:
        for mode in ("malformed", "conflict"):
            self.calls.unlink(missing_ok=True)
            result = self.submit(SBATCH_MODE=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(self.records()), 1)
            self.assertIn("queue", result.stderr)
            self.assertTrue((Path(self.records()[0]["run"]) / "MACE.sbatch.log").is_file())

    def test_later_submission_failure_reports_existing_jobs_and_no_merge(self) -> None:
        result = self.submit(FAIL_JOB="deepmd")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.records()), 2)
        self.assertIn("Already submitted (not cancelled): MACE=1001", result.stderr)

    def test_pinned_list_rechecked_by_compute_preflight(self) -> None:
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        run = Path(self.records()[0]["run"])
        (self.camp / "models/deepmd/dpa2/model_002/model.ckpt.pt").unlink()
        result = subprocess.run(
            ["bash", str(LAUNCHERS / "separation_energy_deepmd.sbatch")],
            env={**self.env, "SEPARATION_CAMPAIGN_ROOT": str(self.camp),
                 "SEPARATION_RUN_DIR": str(run), "INTERFACEFORGE_ROOT": str(ROOT)},
            text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_002", result.stderr)
        self.assertNotIn("module: command not found", result.stderr)

    def test_batch_jobs_use_pinned_models_and_merge_the_same_run(self) -> None:
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        run = Path(self.records()[0]["run"])
        # New exports appear after submission: jobs must still use saved paths.
        self.put(self.mace / "seed_11/mace_model/new_stagetwo.model")
        self.put(self.camp / "models/deepmd/dpa2/model_000/frozen_model.pth")
        self.command("module", "#!/bin/bash\nexit 0\n")
        self.command("nvidia-smi", "#!/bin/bash\nexit 0\n")
        self.command("python", "#!/bin/bash\nexit 0\n")
        activation = self.root / "conda.sh"
        activation.write_text('conda() { return 0; }\n')
        runtime_calls = self.root / "runtime.jsonl"
        self.command("srun", """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['RUNTIME_CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
output = Path(args[args.index('interfaceforge.separation_energy') + 1])
output.mkdir(parents=True, exist_ok=True)
for suffix in ('json', 'csv', 'md', 'png', 'svg', 'pdf'):
    (output / ('separation_energy.' + suffix)).write_text('{}')
""")
        env = {**self.env, "SEPARATION_CAMPAIGN_ROOT": str(self.camp),
               "SEPARATION_RUN_DIR": str(run), "INTERFACEFORGE_ROOT": str(ROOT),
               "MACE_CONDA_SH": str(activation), "RUNTIME_CALLS": str(runtime_calls),
               "INTERFACEFORGE_PYTHON": str(self.bin / "python")}
        for backend in ("mace", "deepmd", "merge"):
            result = subprocess.run(
                ["bash", str(LAUNCHERS / f"separation_energy_{backend}.sbatch")],
                env=env, cwd=self.root, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in runtime_calls.read_text().splitlines()]
        self.assertNotIn(str(self.mace / "seed_11/mace_model/new_stagetwo.model"), calls[0])
        self.assertIn(str(self.mace / "seed_11/mace_model/final_stagetwo.model"), calls[0])
        self.assertIn(str(self.camp / "models/deepmd/dpa2/model_000/model.ckpt.pt"), calls[1])
        self.assertIn(str(run / "stages/mace/separation_energy.json"), calls[2])
        self.assertIn(str(run / "stages/deepmd/separation_energy.json"), calls[2])
        self.assertTrue((run / "separation_energy.pdf").is_file())

    def test_relative_campaign_root_is_normalized_before_export(self) -> None:
        result = self.submit(SEPARATION_CAMPAIGN_ROOT=".")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.records()[0]["root"], str(self.mace))

    def test_merge_submission_failure_records_both_backend_jobs(self) -> None:
        result = self.submit(FAIL_JOB="merge")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MACE=1001", result.stderr)
        self.assertIn("DeePMD=1002", result.stderr)
        run = Path(self.records()[0]["run"])
        self.assertEqual((run / "jobs.tsv").read_text(), "MACE\t1001\nDeePMD\t1002\n")

    def freeze(self, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(LAUNCHERS / "freeze_missing_deepmd_dpa2.sbatch")],
            cwd=self.camp, env={**self.env, "SLURM_SUBMIT_DIR": str(self.camp),
                                "SLURM_ARRAY_TASK_ID": "0", **env}, text=True, capture_output=True,
        )

    def test_freeze_skips_existing_export_without_requiring_checkpoint(self) -> None:
        directory = self.camp / "models/deepmd/dpa2/model_000"
        self.put(directory / "frozen_model.pth", b"existing")
        (directory / "model.ckpt.pt").unlink()
        result = self.freeze()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((directory / "frozen_model.pth").read_bytes(), b"existing")

    def test_freeze_failure_never_publishes_partial_export(self) -> None:
        self.command("module", "#!/bin/bash\nexit 0\n")
        self.command("dp", '#!/bin/bash\nprintf partial > "${@: -1}"\nexit "${DP_FAIL:-0}"\n')
        self.command("python", '#!/bin/bash\nexit "${LOAD_FAIL:-0}"\n')
        directory = self.camp / "models/deepmd/dpa2/model_000"
        for flag in ("DP_FAIL", "LOAD_FAIL"):
            result = self.freeze(**{flag: "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((directory / "frozen_model.pth").exists())
            self.assertEqual([p for p in directory.glob(".freeze.*") if p.is_dir()], [])
        result = self.freeze()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((directory / "frozen_model.pth").read_bytes(), b"partial")

    def test_freeze_rejects_invalid_array_index(self) -> None:
        result = self.freeze(SLURM_ARRAY_TASK_ID="x")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid DPA-2 array index", result.stderr)


if __name__ == "__main__":
    unittest.main()
