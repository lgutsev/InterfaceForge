from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interfaceforge.config import load_campaign
from interfaceforge.errors import ConfigurationError
from interfaceforge.reference_import import (
    activate_reference_profile,
    expand_reference_profile,
    list_reference_profiles,
    load_reference_profile,
    resolve_reference_profiles,
)

_CAMPAIGN = """schema_version: 1
project: {name: t}
profile: profile.yaml
reference: {engine: vasp, inputs: {}}
systems:
  - {id: interface, kind: interface, structure: POSCAR}
dataset: {strategy: grouped, ratios: [0.8, 0.1, 0.1], preserve_raw_forces: true}
models: {deepmd: {enabled: false}, mace: {enabled: false}}
"""
_PROFILE = "name: t\nscheduler: local\njobs: {x: {command: echo hi}}\n"


def _campaign_dir(body: str) -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "profile.yaml").write_text(_PROFILE, encoding="utf-8")
    (directory / "POSCAR").write_text("x", encoding="utf-8")
    (directory / "campaign.yaml").write_text(body, encoding="utf-8")
    return directory


class ReferenceImportTests(unittest.TestCase):
    def test_bundled_sharifi_profile_is_listed_and_loads(self) -> None:
        self.assertIn("sharifi2026", list_reference_profiles())
        profile = load_reference_profile("sharifi2026")
        self.assertEqual(profile["key"], "sharifi2026")
        self.assertEqual(profile["doi"], "10.1016/j.matdes.2026.116095")

    def test_sharifi_expands_to_one_entry_per_quantity(self) -> None:
        entries = expand_reference_profile(load_reference_profile("sharifi2026"))
        quantities = {entry["quantity"] for entry in entries}
        self.assertEqual(quantities, {"work_of_adhesion", "surface_energy"})
        woa = next(e for e in entries if e["quantity"] == "work_of_adhesion")
        self.assertEqual(woa["key"], "sharifi2026")
        self.assertIn("method", woa)  # shared metadata copied onto every entry
        by_term = {
            tuple(sorted(value["match"].items())): value["value_j_per_m2"]
            for value in woa["values"]
        }
        self.assertEqual(
            by_term[(("orientation", "Si3N4(0001)/TiN(111)"), ("termination", "Ti"))], 3.28
        )

    def test_unknown_profile_name_is_a_configuration_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_reference_profile("does-not-exist")

    def test_profile_name_cannot_traverse_directories(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_reference_profile("../secrets")

    def test_profile_can_be_loaded_from_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mine.yaml"
            path.write_text(
                "schema_version: 1\n"
                "key: mine\n"
                "references:\n"
                "  - quantity: work_of_adhesion\n"
                "    values:\n"
                "      - {match: {termination: N}, value_j_per_m2: 1.0}\n",
                encoding="utf-8",
            )
            entries = resolve_reference_profiles([str(path)])
            self.assertEqual(entries[0]["key"], "mine")
            self.assertEqual(entries[0]["values"][0]["value_j_per_m2"], 1.0)

    def test_profile_without_references_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text("schema_version: 1\nkey: bad\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_reference_profile(path)


class ActivateReferenceProfileTests(unittest.TestCase):
    def test_appends_a_validation_block_when_absent_and_reloads(self) -> None:
        campaign = _campaign_dir(_CAMPAIGN) / "campaign.yaml"
        result = activate_reference_profile(campaign, "sharifi2026", write=True)
        self.assertTrue(result["changed"])
        self.assertTrue(result["written"])
        self.assertIn("reference_profiles", campaign.read_text(encoding="utf-8"))
        self.assertTrue(load_campaign(campaign).validation["references"])

    def test_inserts_under_an_existing_validation_block(self) -> None:
        campaign = _campaign_dir(
            _CAMPAIGN + "validation:\n  parity: true\n  work_of_adhesion: true\n"
        ) / "campaign.yaml"
        activate_reference_profile(campaign, "sharifi2026", write=True)
        campaign_data = load_campaign(campaign)
        self.assertEqual(campaign_data.validation["reference_profiles"], ["sharifi2026"])
        self.assertTrue(campaign_data.validation["references"])
        self.assertIn("parity: true", campaign.read_text(encoding="utf-8"))

    def test_extends_an_existing_flow_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary) / "other.yaml"
            other.write_text(
                "schema_version: 1\nkey: other\nreferences:\n"
                "  - quantity: surface_energy\n    values:\n"
                "      - {match: {surface: X}, value_j_per_m2: 1.0}\n",
                encoding="utf-8",
            )
            campaign = _campaign_dir(
                _CAMPAIGN + f"validation:\n  reference_profiles: [{other}]\n"
            ) / "campaign.yaml"
            result = activate_reference_profile(campaign, "sharifi2026", write=True)
            self.assertTrue(result["changed"])
            profiles = load_campaign(campaign).validation["reference_profiles"]
            self.assertEqual(profiles[-1], "sharifi2026")
            self.assertEqual(len(profiles), 2)

    def test_is_idempotent(self) -> None:
        campaign = _campaign_dir(
            _CAMPAIGN + "validation:\n  reference_profiles: [sharifi2026]\n"
        ) / "campaign.yaml"
        before = campaign.read_text(encoding="utf-8")
        result = activate_reference_profile(campaign, "sharifi2026", write=True)
        self.assertTrue(result["already_active"])
        self.assertFalse(result["changed"])
        self.assertEqual(campaign.read_text(encoding="utf-8"), before)

    def test_dry_run_does_not_touch_the_file(self) -> None:
        campaign = _campaign_dir(_CAMPAIGN) / "campaign.yaml"
        before = campaign.read_text(encoding="utf-8")
        result = activate_reference_profile(campaign, "sharifi2026", write=False)
        self.assertEqual(campaign.read_text(encoding="utf-8"), before)
        self.assertFalse(result["written"])
        self.assertIn("sharifi2026", result["resulting_text"])

    def test_inline_validation_mapping_is_refused(self) -> None:
        campaign = _campaign_dir(_CAMPAIGN + "validation: {parity: true}\n") / "campaign.yaml"
        with self.assertRaises(ConfigurationError):
            activate_reference_profile(campaign, "sharifi2026", write=True)

    def test_unknown_profile_is_refused_before_any_edit(self) -> None:
        campaign = _campaign_dir(_CAMPAIGN) / "campaign.yaml"
        before = campaign.read_text(encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            activate_reference_profile(campaign, "nope", write=True)
        self.assertEqual(campaign.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
