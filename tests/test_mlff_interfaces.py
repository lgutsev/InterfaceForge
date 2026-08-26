from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from interfaceforge.campaign import prepare_campaign
from interfaceforge.config import load_campaign
from interfaceforge.errors import SafetyError
from interfaceforge.mlff_interfaces import (
    discover_mlff_interface_sources,
    generate_mlff_interfaces_campaign,
    mass_audit_mlff_interfaces,
    write_throttled_array_launcher,
)

_FAMILIES = ("Real", "Ideal")
_TERMS = ("N_Term", "Ti_Term")
_X_VALUES = ("0", "0.25", "0.5", "0.75", "1.0")


def _write_grid(root: Path, *, term_label: dict[str, str] | None = None) -> Path:
    """A plausible Step2-style source tree: family/term/xVALUE/CONTCAR."""

    source = root / "Step2_450K"
    labels = term_label or {term: term for term in _TERMS}
    for family in _FAMILIES:
        for term in _TERMS:
            for x in _X_VALUES:
                directory = source / family / labels[term] / f"x{x}"
                directory.mkdir(parents=True)
                (directory / "CONTCAR").write_text("dummy contcar\n", encoding="utf-8")
    return source


def _write_profile(path: Path) -> None:
    profile = {
        "name": "loni",
        "scheduler": "slurm",
        "jobs": {
            "vasp_train": {
                "partition": "workq",
                "account": "loni_perovsk27",
                "nodes": 1,
                "ntasks": 64,
                "cpus_per_task": 1,
                "time": "24:00:00",
                "command": "srun -n{ntasks} vasp_std",
            },
            "vasp_train_array": {
                "partition": "workq",
                "account": "loni_perovsk27",
                "nodes": 1,
                "ntasks": 64,
                "cpus_per_task": 1,
                "time": "24:00:00",
            },
        },
    }
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")


class DiscoverySourcesTests(unittest.TestCase):
    def test_standard_layout_matches_all_twenty_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root)

            result = discover_mlff_interface_sources(source, root / "manifest.csv")

            self.assertEqual(result["status_counts"], {"matched": 20})
            self.assertEqual(result["grid_size"], 20)

    def test_alternate_naming_convention_still_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root, term_label={"N_Term": "NTerm", "Ti_Term": "TiTerm"})

            result = discover_mlff_interface_sources(source, root / "manifest.csv")

            self.assertEqual(result["status_counts"], {"matched": 20})

    def test_x_1_point_0_matches_a_bare_x1_directory(self) -> None:
        # A regression guard: x=1.0 formats as "1" via {:g}, which must not
        # fail to match a literal "x1.0" (or "x1") directory name, and must
        # not ambiguously overlap with x=0's "x0" prefix.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src"
            (source / "Real" / "N_Term" / "x1.0").mkdir(parents=True)
            (source / "Real" / "N_Term" / "x1.0" / "CONTCAR").write_text("d\n", encoding="utf-8")
            (source / "Real" / "N_Term" / "x0").mkdir(parents=True)
            (source / "Real" / "N_Term" / "x0" / "CONTCAR").write_text("d\n", encoding="utf-8")

            result = discover_mlff_interface_sources(
                source, root / "manifest.csv", families=("Real",), terms=("N_Term",), x_values=(0.0, 1.0)
            )

            by_x = {row["x"]: row for row in result["rows"]}
            self.assertEqual(by_x["0"]["match_status"], "matched")
            self.assertEqual(by_x["1"]["match_status"], "matched")
            self.assertNotEqual(by_x["0"]["structure_path"], by_x["1"]["structure_path"])

    def test_matches_real_step2_naming_with_bare_x0_and_mixed_separators(self) -> None:
        # The actual observed LONI layout: x=0 (oxygen-free) has no numeric
        # suffix at all ("SiN_TiN_N-term"), oxygen-substituted cells use
        # "..._O_x<value>", N_Term uses underscores and Ti_Term uses hyphens
        # in the leaf name, and the trailing-zero style differs (x1.00 vs
        # x1.0) between the two terms.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Step2_300K"
            leaves = {
                "N_Term": [
                    "SiN_TiN_N-term",
                    "SiN_TiN_N-term_O_x1.00",
                    "SiN_TiN_N-term_O_x0.75",
                    "SiN_TiN_N-term_O_x0.25",
                    "SiN_TiN_N-term_O_x0.5",
                ],
                "Ti_Term": [
                    "SiN-TiN-Ti-term",
                    "SiN-TiN-Ti-term_O_x0.25",
                    "SiN-TiN-Ti-term_O_x1.0",
                    "SiN-TiN-Ti-term_O_x0.5",
                    "SiN-TiN-Ti-term_O_x0.75",
                ],
            }
            for term, names in leaves.items():
                for name in names:
                    directory = source / "Real" / term / name
                    directory.mkdir(parents=True)
                    (directory / "CONTCAR").write_text("dummy\n", encoding="utf-8")
            (source / "Real" / "Ti_Term" / "runvasp.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            result = discover_mlff_interface_sources(
                source, root / "manifest.csv", families=("Real",)
            )

            self.assertEqual(result["status_counts"], {"matched": 10})
            by_key = {(row["term"], row["x"]): row for row in result["rows"]}
            self.assertEqual(
                Path(by_key[("N_Term", "0")]["structure_path"]).parent.name, "SiN_TiN_N-term"
            )
            self.assertEqual(
                Path(by_key[("Ti_Term", "0")]["structure_path"]).parent.name, "SiN-TiN-Ti-term"
            )

    def test_missing_cell_is_reported_not_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src"
            (source / "Real" / "N_Term" / "x0").mkdir(parents=True)
            (source / "Real" / "N_Term" / "x0" / "CONTCAR").write_text("d\n", encoding="utf-8")

            result = discover_mlff_interface_sources(
                source, root / "manifest.csv", families=("Real",), terms=("N_Term",), x_values=(0.0, 0.25)
            )

            statuses = {row["x"]: row["match_status"] for row in result["rows"]}
            self.assertEqual(statuses["0"], "matched")
            self.assertEqual(statuses["0.25"], "missing")

    def test_ambiguous_cell_lists_both_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src"
            for variant in ("run_a", "run_b"):
                d = source / "Real" / "N_Term" / "x0" / variant
                d.mkdir(parents=True)
                (d / "CONTCAR").write_text("d\n", encoding="utf-8")

            result = discover_mlff_interface_sources(
                source, root / "manifest.csv", families=("Real",), terms=("N_Term",), x_values=(0.0,)
            )

            row = result["rows"][0]
            self.assertEqual(row["match_status"], "ambiguous")
            self.assertIn("run_a", row["candidates"])
            self.assertIn("run_b", row["candidates"])


class GenerateCampaignTests(unittest.TestCase):
    def test_refuses_manifest_with_unresolved_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "system_id,family,term,x,structure_path,match_status,candidates\n"
                "real-n_term-x0,Real,N_Term,0,,missing,\n",
                encoding="utf-8",
            )

            with self.assertRaises(SafetyError):
                generate_mlff_interfaces_campaign(
                    manifest, root / "campaign", profile_path=root / "profile.yaml"
                )

    def test_generates_campaign_with_temperature_ramp_and_no_shared_potcar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root)
            discover_mlff_interface_sources(source, root / "manifest.csv")
            _write_profile(root / "profile.yaml")

            result = generate_mlff_interfaces_campaign(
                root / "manifest.csv", root / "VASP_MLFF_Interfaces", profile_path=root / "profile.yaml"
            )

            self.assertEqual(result["systems"], 20)
            campaign_yaml = yaml.safe_load(Path(result["campaign"]).read_text(encoding="utf-8"))
            self.assertEqual(len(campaign_yaml["systems"]), 20)
            self.assertNotIn("POTCAR", campaign_yaml["reference"]["inputs"])
            train_settings = campaign_yaml["stages"]["vasp_mlff"]["train"]
            self.assertEqual(train_settings["temperature"], 300.0)
            self.assertEqual(train_settings["teend"], 600.0)
            incar_text = (Path(result["campaign_root"]) / "inputs" / "INCAR").read_text(encoding="utf-8")
            self.assertIn("ENCUT = 520", incar_text)
            self.assertIn("IVDW = 11", incar_text)


class EndToEndPipelineTests(unittest.TestCase):
    def test_prepare_array_launch_and_audit_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root)
            discover_mlff_interface_sources(source, root / "manifest.csv")
            _write_profile(root / "profile.yaml")
            generate_mlff_interfaces_campaign(
                root / "manifest.csv", root / "VASP_MLFF_Interfaces", profile_path=root / "profile.yaml"
            )
            campaign = load_campaign(root / "VASP_MLFF_Interfaces" / "campaign.yaml")
            prepare_campaign(campaign)

            array_result = write_throttled_array_launcher(campaign, stage="train", concurrency=4)

            self.assertEqual(array_result["leaves"], 20)
            self.assertIn("--array=0-19%4", Path(array_result["launcher"]).read_text(encoding="utf-8"))

            system_id = campaign.systems[0].id
            leaf = campaign.root / "runs" / "vasp" / system_id / "train"
            (leaf / "OUTCAR").write_text(
                "General timing and accounting informations for this job\n", encoding="utf-8"
            )
            (leaf / "OSZICAR").write_text(" 1 T= 300.0 E= -1\n", encoding="utf-8")
            (leaf / "ML_LOGFILE").write_text("STATUS accepted\n", encoding="utf-8")

            audit = mass_audit_mlff_interfaces(campaign)

            self.assertEqual(audit["grid_cells"], 20)
            self.assertEqual(audit["unparsed_runs"], [])
            cell = audit["cells"][system_id]
            self.assertEqual(cell["family"], campaign.systems[0].tags["family"])
            self.assertEqual(cell["term"], campaign.systems[0].tags["term"])
            # 2 terms x 5 x-values per family.
            family_counts = audit["train_health_by_family"][cell["family"]]
            self.assertEqual(sum(family_counts.values()), 10)

    def test_array_launch_requires_prepare_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root)
            discover_mlff_interface_sources(source, root / "manifest.csv")
            _write_profile(root / "profile.yaml")
            generate_mlff_interfaces_campaign(
                root / "manifest.csv", root / "VASP_MLFF_Interfaces", profile_path=root / "profile.yaml"
            )
            campaign = load_campaign(root / "VASP_MLFF_Interfaces" / "campaign.yaml")

            with self.assertRaises(SafetyError):
                write_throttled_array_launcher(campaign, stage="train")

    def test_audit_requires_prepare_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_grid(root)
            discover_mlff_interface_sources(source, root / "manifest.csv")
            _write_profile(root / "profile.yaml")
            generate_mlff_interfaces_campaign(
                root / "manifest.csv", root / "VASP_MLFF_Interfaces", profile_path=root / "profile.yaml"
            )
            campaign = load_campaign(root / "VASP_MLFF_Interfaces" / "campaign.yaml")

            with self.assertRaises(SafetyError):
                mass_audit_mlff_interfaces(campaign)


if __name__ == "__main__":
    unittest.main()
