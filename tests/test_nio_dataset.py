"""Canonical NiO dataset export: discovery, frame QC, leakage-safe splits, provenance."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from ase.io import read
from nio_fixture import CELL, build_nio_tree, write_poscar, write_run

from interfaceforge.cli import main as cli_main
from interfaceforge.errors import ConfigurationError, SafetyError
from interfaceforge.nio_dataset import (
    ExportConfig,
    analyze_trajectory,
    assign_groups,
    discover_report,
    discover_trajectories,
    export_dataset,
    leakage_report,
    parse_nio_case,
    plan_export,
    split_groups,
    split_hash,
    verify_dataset,
)
from interfaceforge.training import validate_deepmd_dataset

CASE = "NiO_m110_Big_U46_OH50_clustered_dissoc"


def _by_id(trajectories, suffix):
    return next(item for item in trajectories if item.trajectory_id.endswith(suffix))


class CaseNameParsingTests(unittest.TestCase):
    def test_parses_notebook_naming_convention(self) -> None:
        bare = parse_nio_case("NiO_m110_Big_U46")
        self.assertEqual(bare["coverage_pct"], 0)
        self.assertIsNone(bare["ligand"])
        self.assertEqual(bare["surface"], "NiO_m110_Big_U46")
        full = parse_nio_case("NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary")
        self.assertEqual(
            {key: full[key] for key in ("coverage_pct", "pattern", "motif", "ligand", "anchor")},
            {"coverage_pct": 50, "pattern": "clustered", "motif": "capped", "ligand": "Me4PACz", "anchor": "boundary"},
        )
        self.assertEqual(full["surface"], "NiO_m110_Big_U46_OH50_clustered_capped")
        hyphenated = parse_nio_case("NiO_m110_Big_U46_OH100_full_dissoc_MeO-2PACz_hbond")
        self.assertEqual(hyphenated["ligand"], "MeO2PACz")
        self.assertEqual(hyphenated["anchor"], "hbond")
        pristine_ligand = parse_nio_case("NiO_m110_Big_U46_DCZ-4P")
        self.assertEqual((pristine_ligand["coverage_pct"], pristine_ligand["ligand"]), (0, "DCZ4P"))

    def test_unknown_tokens_are_reported_not_guessed(self) -> None:
        parsed = parse_nio_case("NiO_m110_Big_U46_OH25_scattered_capped_NewLigand_bare")
        self.assertIsNone(parsed["ligand"])
        self.assertIn("NewLigand", parsed["unparsed_tokens"])
        self.assertEqual(parsed["parse"], "partial")


class DiscoveryTests(unittest.TestCase):
    def test_stage_temperature_case_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head = build_nio_tree(Path(temporary) / "NiO_head", cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))])
            rng = np.random.default_rng(0)
            start = rng.random((14, 3)) * 5
            for hidden in ("Step1/OH50/archive/old", "Step1/OH50/X_disabled", "Step2_300K/OH50/restart_archive_1"):
                write_run(
                    head / hidden,
                    species=("H", "Ni", "O"),
                    counts=(4, 4, 6),
                    start_positions=start,
                    n_steps=2,
                    rng=rng,
                    tebeg=300,
                )
            # A conservative Step1 leaf carries a precondition/ child; the leaf is still a trajectory.
            write_run(
                head / "Step1" / "OH50" / CASE / "precondition",
                species=("H", "Ni", "O"),
                counts=(4, 4, 6),
                start_positions=start,
                n_steps=1,
                rng=rng,
                tebeg=300,
            )
            found = discover_trajectories([head])
            ids = [item.trajectory_id for item in found]
            self.assertEqual(len(ids), 4, ids)
            self.assertFalse(any(token in text for text in ids for token in ("archive", "X_disabled", "precondition")))
            step1 = _by_id(found, f"Step1/OH50/{CASE}")
            self.assertEqual((step1.stage, step1.case, step1.temperature_k), ("Step1", f"OH50/{CASE}", 300.0))
            self.assertEqual(step1.temperature_source, "INCAR TEEND")
            step2 = _by_id(found, f"Step2_450K/OH50/{CASE}")
            self.assertEqual(
                (step2.stage, step2.temperature_k, step2.temperature_source), ("Step2", 450.0, "stage directory name")
            )
            self.assertEqual(step2.chemistry["coverage_pct"], 50)

    def test_gzipped_outcar_is_discovered_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head = build_nio_tree(
                Path(temporary) / "h",
                cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))],
                temperatures=(300,),
                special={f"Step2_300K/{CASE}": {"gz": True}},
            )
            found = discover_trajectories([head])
            step2 = _by_id(found, f"Step2_300K/OH50/{CASE}")
            self.assertEqual(step2.outcar.name, "OUTCAR.gz")
            analysis = analyze_trajectory(step2, ExportConfig())
            self.assertEqual(analysis.frames_parsed, 8)


class FrameQualityTests(unittest.TestCase):
    def _analysis(self, special: dict, stage_key: str, **config):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        head = build_nio_tree(
            Path(temporary.name) / "h",
            cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))],
            temperatures=(300,),
            special={stage_key: special},
        )
        trajectory = _by_id(discover_trajectories([head]), f"{stage_key.split('/')[0]}/OH50/{CASE}")
        return analyze_trajectory(trajectory, ExportConfig(**config))

    def test_truncated_final_step_is_reported_and_run_marked_incomplete(self) -> None:
        analysis = self._analysis({"truncate_last": True}, f"Step2_300K/{CASE}")
        self.assertEqual(analysis.ionic_blocks, 8)
        self.assertEqual(analysis.frames_parsed, 7)
        self.assertEqual(analysis.status, "incomplete")
        self.assertIn({"source_frame": 7, "reason": "truncated ionic step (no energy block)"}, analysis.rejected)
        self.assertTrue(any("truncated" in warning for warning in analysis.warnings))

    def test_scf_unconverged_steps_are_rejected(self) -> None:
        analysis = self._analysis({"scf_ceiling_steps": (1, 3)}, f"Step2_300K/{CASE}", use_step2_sample=False)
        rejected = {row["source_frame"] for row in analysis.rejected}
        self.assertEqual(rejected, {0, 2})
        self.assertEqual(analysis.scf_unconverged_steps, 2)
        self.assertEqual([frame.source_frame for frame in analysis.frames], [1, 3, 4, 5, 6, 7])
        kept = self._analysis(
            {"scf_ceiling_steps": (1, 3)}, f"Step2_300K/{CASE}", use_step2_sample=False, reject_scf_unconverged=False
        )
        self.assertEqual(len(kept.frames), 8)

    def test_frames_after_temperature_runaway_are_rejected(self) -> None:
        analysis = self._analysis({"runaway_from": 5}, f"Step2_300K/{CASE}", use_step2_sample=False)
        self.assertEqual(analysis.first_bad_step, 5)
        self.assertEqual([frame.source_frame for frame in analysis.frames], [0, 1, 2, 3])
        self.assertTrue(any("temperature runaway" in warning for warning in analysis.warnings))

    def test_energy_reference_excursion_is_reported_not_rejected(self) -> None:
        analysis = self._analysis({"energy_jump_from": 4}, f"Step2_300K/{CASE}", use_step2_sample=False)
        self.assertEqual(len(analysis.frames), 8)
        self.assertTrue(any("reported, not rejected" in warning for warning in analysis.warnings))

    def test_non_finite_energy_is_rejected(self) -> None:
        analysis = self._analysis({"nan_energy_at": 2}, f"Step2_300K/{CASE}", use_step2_sample=False)
        reasons = {row["source_frame"]: row["reason"] for row in analysis.rejected}
        self.assertIn("non-finite energy", reasons[2])

    def test_atom_order_disagreeing_with_poscar_rejects_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head = build_nio_tree(
                Path(temporary) / "h", cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))], temperatures=(300,)
            )
            run = head / "Step2_300K" / "OH50" / CASE
            write_poscar(run / "POSCAR", ("Ni", "H", "O"), (4, 4, 6), CELL, np.zeros((14, 3)))
            trajectory = _by_id(discover_trajectories([head]), f"Step2_300K/OH50/{CASE}")
            analysis = analyze_trajectory(trajectory, ExportConfig())
            self.assertEqual(analysis.frames, [])
            self.assertEqual(analysis.status, "no_usable_frames")
            self.assertTrue(all("POSCAR" in row["reason"] for row in analysis.rejected))

    def test_step2_sample_indices_are_honoured_else_stride(self) -> None:
        sampled = self._analysis({}, f"Step2_300K/{CASE}")
        self.assertEqual(sampled.selection, "step2_sample")
        self.assertEqual([frame.source_frame for frame in sampled.frames], [0, 2, 4, 6])
        strided = self._analysis({}, f"Step2_300K/{CASE}", use_step2_sample=False, stride=3)
        self.assertEqual(strided.selection, "stride")
        self.assertEqual([frame.source_frame for frame in strided.frames], [0, 3, 6])

    def test_constraints_come_from_poscar_selective_dynamics(self) -> None:
        analysis = self._analysis({}, f"Step2_300K/{CASE}")
        self.assertEqual(analysis.constraint_source, "POSCAR selective dynamics")
        self.assertEqual(int(analysis.move_mask.sum()), 12)
        self.assertEqual(list(analysis.move_mask[:2]), [0, 0])


class SplitTests(unittest.TestCase):
    def test_temperature_series_of_one_case_never_spans_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head = build_nio_tree(Path(temporary) / "h")
            plan = plan_export([head], ExportConfig(stages=("Step1", "Step2")))
            by_case: dict[str, set[str]] = {}
            for analysis in plan.usable:
                by_case.setdefault(analysis.trajectory.case, set()).add(
                    plan.trajectory_split[analysis.trajectory.trajectory_id]
                )
            self.assertTrue(all(len(splits) == 1 for splits in by_case.values()), by_case)
            self.assertFalse(plan.leakage["leakage_detected"])
            self.assertEqual(len(plan.group_split), 6)
            self.assertTrue(all(plan.leakage[key] == [] for key in ("problems", "cases_spanning_splits")))

    def test_lineage_is_linked_across_roots_with_different_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            head = build_nio_tree(root / "h", cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))], temperatures=(300,))
            # Move the Step2 tree to a second project root and rename the case directory:
            # only the CONTCAR -> POSCAR hand-off still links it to its Step1 parent.
            other = root / "renamed_head" / "Step2_300K" / "copy"
            other.parent.mkdir(parents=True)
            (head / "Step2_300K" / "OH50" / CASE).rename(other)
            trajectories = discover_trajectories([head, root / "renamed_head"])
            config = ExportConfig(stages=("Step1", "Step2"))
            analyses = [analyze_trajectory(item, config) for item in trajectories]
            groups = assign_groups(analyses)
            self.assertEqual(len(set(groups.values())), 1, groups)

    def test_leakage_report_flags_a_group_in_two_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head = build_nio_tree(
                Path(temporary) / "h", cases=[("OH50", CASE, ("H", "Ni", "O"), (4, 4, 6))], temperatures=(300, 450)
            )
            plan = plan_export([head], ExportConfig())
            ids = sorted(plan.trajectory_split)
            forced = {ids[0]: "train", ids[1]: "test"}
            report = leakage_report(plan.usable, plan.groups, forced)
            self.assertTrue(report["leakage_detected"])
            self.assertIn("group spans splits", {problem["check"] for problem in report["problems"]})

    def test_balanced_split_is_deterministic_and_fills_every_split(self) -> None:
        frames = {f"g{index:02d}": 10 + index for index in range(20)}
        strata = {group: ("a" if index % 2 else "b") for index, group in enumerate(frames)}
        first = split_groups(frames, strata, (0.8, 0.1, 0.1), seed=7)
        second = split_groups(dict(reversed(list(frames.items()))), strata, (0.8, 0.1, 0.1), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "valid", "test"})
        totals = {
            split: sum(frames[group] for group, value in first.items() if value == split)
            for split in ("train", "valid", "test")
        }
        self.assertGreater(totals["train"] / sum(totals.values()), 0.65)
        self.assertNotEqual(first, split_groups(frames, strata, (0.8, 0.1, 0.1), seed=8))

    def test_hash_split_is_stable_when_groups_are_added(self) -> None:
        frames = {f"g{index:02d}": 5 for index in range(30)}
        before = split_groups(frames, {}, (0.8, 0.1, 0.1), seed=3, method="hash")
        frames["new_case"] = 5
        after = split_groups(frames, {}, (0.8, 0.1, 0.1), seed=3, method="hash")
        self.assertEqual({group: after[group] for group in before}, before)

    def test_split_hash_is_order_independent(self) -> None:
        pairs = [("a:1", "train"), ("b:2", "test")]
        self.assertEqual(split_hash(pairs), split_hash(list(reversed(pairs))))
        self.assertNotEqual(split_hash(pairs), split_hash([("a:1", "test"), ("b:2", "test")]))

    def test_config_validation(self) -> None:
        with self.assertRaises(ConfigurationError):
            ExportConfig(stages=("Step9",))
        with self.assertRaises(ConfigurationError):
            ExportConfig(group_by="random-frame")
        with self.assertRaises(ConfigurationError):
            ExportConfig(ratios=(1, -1, 0))
        with self.assertRaises(ConfigurationError):
            ExportConfig(stride=0)
        self.assertEqual(ExportConfig(ratios=(8, 1, 1)).ratios, (0.8, 0.1, 0.1))


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.head = build_nio_tree(
            self.root / "NiO_head",
            special={
                "Step2_600K/NiO_m110_Big_U46_OH75_scattered_capped": {"truncate_last": True},
                f"Step2_450K/{CASE}": {"scf_ceiling_steps": (3,)},
            },
        )

    def test_export_writes_one_canonical_dataset_for_every_backend(self) -> None:
        payload = export_dataset([self.head], self.root / "canonical", ExportConfig())
        out = Path(payload["output_root"])
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "interfaceforge-canonical-dataset")
        self.assertEqual(manifest["type_map"], ["C", "H", "N", "Ni", "O", "P"])
        self.assertFalse(payload["leakage_detected"])
        self.assertEqual(payload["trajectories_exported"], 17)
        self.assertEqual(set(manifest["backends"]), {"mace", "deepmd", "nequip"})
        self.assertEqual(
            manifest["backends"]["nequip"]["key_mapping"], {"REF_energy": "total_energy", "REF_forces": "forces"}
        )
        self.assertTrue(manifest["interfaceforge_commit"]["source"])

        frames = read(out / "test.extxyz", index=":")
        info = frames[0].info
        for key in (
            "frame_id",
            "IF_leaf",
            "source_path",
            "source_frame",
            "IF_stage",
            "IF_temperature_k",
            "IF_case",
            "IF_group",
            "IF_formula",
        ):
            self.assertIn(key, info)
        self.assertEqual(info["frame_id"], f"{info['IF_leaf']}:{info['source_frame']}")
        self.assertEqual(len(frames[0].constraints[0].get_indices()), 2)
        self.assertTrue(np.any(np.abs(frames[0].arrays["REF_forces"][:2]) > 0), "frozen-atom forces must stay raw")

        type_map, systems = validate_deepmd_dataset(out / "deepmd")
        self.assertEqual(type_map, manifest["type_map"])
        system = Path(systems["test"][0])
        self.assertTrue((system / "move_mask.npy").is_file())
        meta = json.loads((system / "system_meta.json").read_text())
        self.assertIn(meta["split"], {"test"})
        with (system / "frame_map.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["relative_leaf"], meta["run_id"])

        report = verify_dataset(out)
        self.assertTrue(report["valid"], report["problems"])
        self.assertTrue(all(item["exact_membership"] for item in report["extxyz_deepmd_membership"].values()))

        with (out / "trajectories.csv").open() as handle:
            rows = {row["trajectory_id"]: row for row in csv.DictReader(handle)}
        truncated = rows["NiO_head/Step2_600K/OH75/NiO_m110_Big_U46_OH75_scattered_capped"]
        self.assertEqual(truncated["status"], "incomplete")
        self.assertEqual(rows["NiO_head/Step1/OH0/NiO_m110_Big_U46"]["status"], "stage_not_selected")
        with (out / "rejected_frames.csv").open() as handle:
            rejected = list(csv.DictReader(handle))
        self.assertTrue(any("SCF not converged" in row["reason"] for row in rejected))

    def test_export_is_deterministic_and_verify_detects_tampering(self) -> None:
        first = export_dataset([self.head], self.root / "a", ExportConfig())
        second = export_dataset([self.head], self.root / "b", ExportConfig())
        self.assertEqual(first["dataset_hash"], second["dataset_hash"])
        self.assertEqual(first["split_hash"], second["split_hash"])
        reseeded = export_dataset([self.head], self.root / "c", ExportConfig(seed=99))
        self.assertEqual(
            reseeded["frame_counts"]["train"] + reseeded["frame_counts"]["valid"] + reseeded["frame_counts"]["test"],
            sum(first["frame_counts"].values()),
        )
        target = self.root / "a" / "valid.extxyz"
        target.write_text(target.read_text().replace("REF_energy=-", "REF_energy=-1", 1))
        report = verify_dataset(self.root / "a")
        self.assertFalse(report["valid"])
        self.assertTrue(any("hash mismatch valid.extxyz" in problem for problem in report["problems"]))

    def test_export_refuses_unsafe_outputs(self) -> None:
        with self.assertRaises(SafetyError):
            export_dataset([self.head], self.head / "inside", ExportConfig())
        (self.root / "busy").mkdir()
        (self.root / "busy" / "x").write_text("keep")
        with self.assertRaises(SafetyError):
            export_dataset([self.head], self.root / "busy", ExportConfig())

    def test_explicit_type_map_must_cover_the_data(self) -> None:
        with self.assertRaisesRegex(SafetyError, "missing elements"):
            plan_export([self.head], ExportConfig(type_map=("Ni", "O")))

    def test_dataset_is_archivable_with_dedupe_and_materializes_exactly(self) -> None:
        from interfaceforge.packaging import materialize_dataset, pack_dataset_archive, verify_package

        out = Path(export_dataset([self.head], self.root / "canonical", ExportConfig())["output_root"])
        archive = pack_dataset_archive(out, self.root / "backup.zip", dedupe=True)
        self.assertTrue(verify_package(archive["archive"])["valid"])
        rebuilt = self.root / "rebuilt"
        materialize_dataset(out, rebuilt)
        original = read(out / "train.extxyz", index=":")
        again = read(rebuilt / "train.extxyz", index=":")
        self.assertEqual(len(original), len(again))
        self.assertTrue(np.array_equal(original[0].arrays["REF_forces"], again[0].arrays["REF_forces"]))

    def test_cli_discover_export_verify(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(
                cli_main(
                    [
                        "dataset",
                        "discover",
                        str(self.head),
                        "--stages",
                        "Step1",
                        "Step2",
                        "--ratios",
                        "0.7",
                        "0.15",
                        "0.15",
                    ]
                ),
                0,
            )
        summary = json.loads(buffer.getvalue())["summary"]
        self.assertEqual(summary["stage_counts"], {"Step1": 6, "Step2": 18})
        self.assertIn("rejection_reasons", summary)
        output = self.root / "cli_dataset"
        with redirect_stdout(StringIO()):
            self.assertEqual(
                cli_main(["dataset", "export", str(self.head), "--output", str(output), "--seed", "11"]), 0
            )
        buffer = StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(cli_main(["dataset", "verify", str(output)]), 0)
        self.assertTrue(json.loads(buffer.getvalue())["valid"])
        report = discover_report([self.head], ExportConfig(seed=11))
        self.assertEqual(
            json.loads((output / "manifest.json").read_text())["summary"]["frames_per_split"],
            report["summary"]["frames_per_split"],
        )


if __name__ == "__main__":
    unittest.main()
