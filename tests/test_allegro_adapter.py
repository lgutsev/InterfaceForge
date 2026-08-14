from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from interfaceforge.allegro import (
    MIN_LAMMPS_DATE,
    _parse_lammps_date,
    allegro_lammps_preflight,
    generate_allegro_training,
)
from interfaceforge.config import Campaign


class AllegroAdapterTests(unittest.TestCase):
    def test_lammps_date_parser(self) -> None:
        self.assertEqual(_parse_lammps_date("LAMMPS (10 Sep 2025)"), MIN_LAMMPS_DATE)
        self.assertIsNone(_parse_lammps_date("LAMMPS unknown"))

    def test_preflight_detects_modern_pair_allegro_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lmp = root / "lmp"
            lmp.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<'EOF'\n"
                "LAMMPS (10 Dec 2025)\n"
                "Installed packages: KOKKOS\n"
                "Pair styles: allegro lj/cut\n"
                "EOF\n",
                encoding="utf-8",
            )
            lmp.chmod(0o750)
            model = root / "model.nequip.pt2"
            model.write_text("placeholder", encoding="utf-8")
            payload = allegro_lammps_preflight(str(lmp), str(model))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["release"], "2025-12-10")

    def test_preflight_rejects_old_lammps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lmp = root / "lmp"
            lmp.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'LAMMPS (29 Aug 2024)'\n"
                "echo 'KOKKOS allegro'\n",
                encoding="utf-8",
            )
            lmp.chmod(0o750)
            payload = allegro_lammps_preflight(str(lmp))
            self.assertFalse(payload["passed"])

    def test_generation_is_dependency_light_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = root / "datasets" / "canonical"
            datasets.mkdir(parents=True)
            for split in ("train", "valid", "test"):
                (datasets / f"{split}.extxyz").write_text("", encoding="utf-8")
            profile_path = root / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "local",
                        "scheduler": "local",
                        "jobs": {
                            "allegro_gpu": {"command": "nequip-train"},
                            "allegro_lammps": {"command": "lmp"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            campaign = Campaign(
                path=root / "campaign.yaml",
                root=root,
                name="allegro-test",
                description="",
                profile_path=profile_path,
                systems=(),
                reference={"engine": "vasp"},
                stages={},
                dataset={"type_map": ["Ni", "O"]},
                models={
                    "allegro": {
                        "enabled": True,
                        "profile": "allegro_gpu",
                        "lammps_profile": "allegro_lammps",
                    }
                },
                active_learning={},
                exploration={},
                validation={},
                raw={},
            )
            manifest = generate_allegro_training(campaign)
            self.assertEqual(manifest["type_names"], ["Ni", "O"])
            config = yaml.safe_load((root / "models" / "allegro" / "config.yaml").read_text())
            self.assertEqual(config["training_module"]["model"]["_target_"], "allegro.model.AllegroModel")
            build = (root / "models" / "allegro" / "lammps" / "build_pair_allegro.sh").read_text()
            self.assertIn("PKG_KOKKOS=ON", build)
            self.assertIn("NEQUIP_AOT_COMPILE=ON", build)
            launcher = (root / "models" / "allegro" / "lammps" / "run_lammps.slurm").read_text()
            self.assertIn("lammps-preflight", launcher)


if __name__ == "__main__":
    unittest.main()
