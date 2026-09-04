from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "launch_scripts"


class SeparationEnergyLauncherTests(unittest.TestCase):
    def test_launchers_have_valid_bash_syntax_and_are_executable(self) -> None:
        names = (
            "separation_energy_mace.sbatch",
            "separation_energy_deepmd.sbatch",
            "separation_energy_merge.sbatch",
            "freeze_missing_deepmd_dpa2.sbatch",
            "submit_separation_energy.sh",
        )
        for name in names:
            path = LAUNCHERS / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(os.access(path, os.X_OK), name)
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_gpu_launchers_keep_compiled_backends_isolated(self) -> None:
        mace = (LAUNCHERS / "separation_energy_mace.sbatch").read_text(encoding="utf-8")
        deepmd = (LAUNCHERS / "separation_energy_deepmd.sbatch").read_text(encoding="utf-8")

        self.assertIn("conda activate /project/lgutsev/env/mace_env", mace)
        self.assertNotIn("module load \"$DEEPMD_MODULE\"", mace)
        self.assertIn("! -name '*_compiled.model'", mace)
        self.assertIn("has no stage-two-named model", mace)
        self.assertIn("module load \"$DEEPMD_MODULE\"", deepmd)
        self.assertNotIn("conda activate /project/lgutsev/env/mace_env", deepmd)
        self.assertIn("--json-only", mace)
        self.assertIn("--json-only", deepmd)
        self.assertIn("model.ckpt.pt", deepmd)
        self.assertIn('if [[ -s "$frozen" ]]', deepmd)

    def test_mace_committee_root_matches_the_project_convention(self) -> None:
        # `iface mlip-progress` / `iface package campaign` both default the MACE
        # committee root to models/mace_committee_520eV -- the committee is
        # *not* directly under models/mace_committee/. Both scripts that look
        # for a seed's trained model must agree, and honour an override.
        submitter = (LAUNCHERS / "submit_separation_energy.sh").read_text(encoding="utf-8")
        mace = (LAUNCHERS / "separation_energy_mace.sbatch").read_text(encoding="utf-8")
        for text in (submitter, mace):
            self.assertIn(
                'MACE_COMMITTEE_ROOT="${MACE_COMMITTEE_ROOT:-$CAMP/models/mace_committee_520eV}"',
                text,
            )
            self.assertIn('seed_dir="$MACE_COMMITTEE_ROOT/mace_committee/seed_${seed}"', text)
            self.assertNotIn('seed_dir="$CAMP/models/mace_committee/seed_${seed}"', text)
        self.assertIn("MACE_COMMITTEE_ROOT=$MACE_COMMITTEE_ROOT", submitter)

    def test_submitter_attaches_merge_with_afterok(self) -> None:
        submitter = (LAUNCHERS / "submit_separation_energy.sh").read_text(encoding="utf-8")
        merge = (LAUNCHERS / "separation_energy_merge.sbatch").read_text(encoding="utf-8")

        self.assertIn('--dependency="afterok:${mace_id}:${deepmd_id}"', submitter)
        self.assertIn("INTERFACEFORGE_ROOT=$REPO_ROOT", submitter)
        self.assertIn("final standalone job-id line", submitter)
        self.assertIn("model.ckpt.pt", submitter)
        self.assertIn("expected one usable MACE model", submitter)
        self.assertIn("--merge-json", merge)
        self.assertNotIn("#SBATCH --gres=gpu", merge)

    def test_freeze_recovery_is_an_idempotent_array_job(self) -> None:
        freeze = (LAUNCHERS / "freeze_missing_deepmd_dpa2.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn("#SBATCH --array=0-3%1", freeze)
        self.assertIn('dp --pt freeze -c "$checkpoint" -o "$frozen"', freeze)
        self.assertIn("already has a frozen model; leaving it unchanged", freeze)
        self.assertIn("DeepPot(sys.argv[1])", freeze)

    def test_submitter_parses_loni_accounting_output_and_accepts_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            (campaign / "campaign.yaml").write_text("name: test\n", encoding="utf-8")
            for termination in ("N_term_dft", "Ti_term_dft"):
                directory = campaign / "adhesion" / termination
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text("{}\n", encoding="utf-8")
            for seed in (11, 23, 37, 53):
                directory = (
                    campaign
                    / "models"
                    / "mace_committee_520eV"
                    / "mace_committee"
                    / f"seed_{seed}"
                    / "mace_model"
                )
                directory.mkdir(parents=True)
                (directory / "final.model").write_bytes(b"model")
            for member in range(4):
                directory = campaign / "models" / "deepmd" / "dpa2" / f"model_{member:03d}"
                directory.mkdir(parents=True)
                (directory / "model.ckpt.pt").write_bytes(b"checkpoint")

            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            calls = Path(tmp) / "sbatch.calls"
            sbatch = fake_bin / "sbatch"
            sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
                "echo 'sbatch: 4406304.39 SUs available in loni_perovsk27'\n"
                "case \"$*\" in\n"
                "  *separation_energy_mace.sbatch*) echo 1001 ;;\n"
                "  *separation_energy_deepmd.sbatch*) echo 1002 ;;\n"
                "  *separation_energy_merge.sbatch*) echo 1003 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            sbatch.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "FAKE_SBATCH_LOG": str(calls),
            }

            result = subprocess.run(
                [str(LAUNCHERS / "submit_separation_energy.sh")],
                cwd=campaign,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("MACE:   1001", result.stdout)
            self.assertIn("DeePMD: 1002", result.stdout)
            self.assertIn("merge:  1003 (afterok:1001:1002)", result.stdout)
            self.assertIn("SUs available", result.stderr)
            self.assertIn("--dependency=afterok:1001:1002", calls.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
