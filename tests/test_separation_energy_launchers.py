from __future__ import annotations

import os
import subprocess
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
        self.assertIn("module load \"$DEEPMD_MODULE\"", deepmd)
        self.assertNotIn("conda activate /project/lgutsev/env/mace_env", deepmd)
        self.assertIn("--json-only", mace)
        self.assertIn("--json-only", deepmd)

    def test_submitter_attaches_merge_with_afterok(self) -> None:
        submitter = (LAUNCHERS / "submit_separation_energy.sh").read_text(encoding="utf-8")
        merge = (LAUNCHERS / "separation_energy_merge.sbatch").read_text(encoding="utf-8")

        self.assertIn('--dependency="afterok:${mace_id}:${deepmd_id}"', submitter)
        self.assertIn("INTERFACEFORGE_ROOT=$REPO_ROOT", submitter)
        self.assertIn("--merge-json", merge)
        self.assertNotIn("#SBATCH --gres=gpu", merge)


if __name__ == "__main__":
    unittest.main()
