from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from interfaceforge.cli import build_parser
from interfaceforge.errors import DependencyError, SafetyError
from interfaceforge.intermat import generate_intermat_interfaces, intermat_status


class FakeAtoms:
    def __init__(self, payload: dict):
        self.payload = payload
        self.lattice_mat = np.asarray(payload["lattice_mat"], dtype=float)
        self.frac_coords = np.asarray(payload["coords"], dtype=float)
        self.elements = list(payload["elements"])
        self.num_atoms = len(self.elements)

    @classmethod
    def from_poscar(cls, path: str):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(payload)

    def write_poscar(self, path: str):
        Path(path).write_text(json.dumps(self.payload, sort_keys=True) + "\n", encoding="utf-8")


class FakeInterfaceCombi:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeInterfaceCombi.last_kwargs = kwargs

    def generate(self):
        lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]]
        first = {
            "lattice_mat": lattice,
            "coords": [[0.0, 0.0, 0.25], [0.5, 0.5, 0.75]],
            "elements": ["A", "B"],
        }
        periodic_duplicate = {
            "lattice_mat": lattice,
            "coords": [[1.0, 0.0, 0.25], [0.5, 0.5, 0.75]],
            "elements": ["A", "B"],
        }
        second = {
            "lattice_mat": lattice,
            "coords": [[0.25, 0.0, 0.25], [0.5, 0.5, 0.75]],
            "elements": ["A", "B"],
        }

        def candidate(payload: dict, name: str, mismatch: float) -> dict:
            return {
                "generated_interface": payload,
                "interface_name": name,
                "mismatch_u": np.float64(mismatch),
                "mismatch_v": np.float64(-mismatch),
                "mismatch_angle": np.float64(0.1),
                "area1": np.float64(25.0),
                "area2": np.float64(25.1),
            }

        return [
            candidate(first, "Interface-film-sub_seperation_2.5_disp_0.0_0.0", 0.01),
            candidate(periodic_duplicate, "Interface-film-sub_seperation_2.5_disp_1.0_0.0", 0.01),
            candidate(second, "Interface-film-sub_seperation_2.5_disp_0.25_0.0", 0.02),
        ]


def write_bulk(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                "coords": [[0.0, 0.0, 0.0]],
                "elements": ["A"],
            }
        ),
        encoding="utf-8",
    )


class InterMatAdapterTests(unittest.TestCase):
    def test_status_is_safe_without_optional_dependency(self) -> None:
        result = intermat_status()
        self.assertIn("available", result)
        self.assertEqual(result["adapter"], "interfaceforge.intermat")

    def test_cli_parses_status_and_generate(self) -> None:
        parser = build_parser()
        status = parser.parse_args(["intermat", "status"])
        self.assertEqual(status.intermat_command, "status")
        generate = parser.parse_args(
            [
                "intermat",
                "generate",
                "film.vasp",
                "substrate.vasp",
                "generated",
                "--film-miller",
                "1",
                "1",
                "0",
                "--separation",
                "2.5",
                "--separation",
                "3.0",
            ]
        )
        self.assertEqual(generate.film_miller, [1, 1, 0])
        self.assertEqual(generate.separation, [2.5, 3.0])

    def test_generate_deduplicates_and_exports_campaign_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            film = root / "film.vasp"
            substrate = root / "substrate.vasp"
            write_bulk(film)
            write_bulk(substrate)
            output = root / "generated"
            runtime = {"Atoms": FakeAtoms, "InterfaceCombi": FakeInterfaceCombi}
            with mock.patch("interfaceforge.intermat._runtime", return_value=runtime):
                result = generate_intermat_interfaces(
                    film,
                    substrate,
                    output,
                    displacement_interval=0.25,
                )
            self.assertEqual(result["raw_candidates"], 3)
            self.assertEqual(result["unique_candidates"], 2)
            self.assertEqual(result["periodic_duplicates_removed"], 1)
            manifest = json.loads((output / "intermat_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["candidates"]), 2)
            self.assertEqual(manifest["candidates"][1]["fractional_displacement"], [0.25, 0.0])
            fragment = yaml.safe_load((output / "campaign_fragment.yaml").read_text(encoding="utf-8"))
            self.assertEqual(fragment["systems"][0]["kind"], "interface")
            self.assertTrue((output / fragment["systems"][0]["structure"]).is_file())
            self.assertFalse(FakeInterfaceCombi.last_kwargs["apply_strain"])

    def test_candidate_cap_fails_before_loading_intermat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            film = root / "film.vasp"
            substrate = root / "substrate.vasp"
            write_bulk(film)
            write_bulk(substrate)
            with mock.patch("interfaceforge.intermat._runtime") as runtime:
                with self.assertRaisesRegex(SafetyError, "above --max-candidates"):
                    generate_intermat_interfaces(
                        film,
                        substrate,
                        root / "generated",
                        displacement_interval=0.1,
                        max_candidates=100,
                    )
            runtime.assert_not_called()

    def test_force_refuses_unrecognized_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            film = root / "film.vasp"
            substrate = root / "substrate.vasp"
            write_bulk(film)
            write_bulk(substrate)
            output = root / "generated"
            output.mkdir()
            (output / "user-data.txt").write_text("preserve me\n", encoding="utf-8")
            with self.assertRaisesRegex(SafetyError, "not a recognized"):
                generate_intermat_interfaces(film, substrate, output, force=True)
            self.assertTrue((output / "user-data.txt").is_file())

    def test_dependency_failure_does_not_leave_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            film = root / "film.vasp"
            substrate = root / "substrate.vasp"
            write_bulk(film)
            write_bulk(substrate)
            output = root / "generated"
            with mock.patch(
                "interfaceforge.intermat._runtime",
                side_effect=DependencyError("missing"),
            ):
                with self.assertRaises(DependencyError):
                    generate_intermat_interfaces(film, substrate, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
