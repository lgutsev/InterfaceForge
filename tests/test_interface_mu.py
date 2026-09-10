from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interfaceforge.errors import SafetyError
from interfaceforge.interface_mu import (
    audit_molecular_spin,
    gamma_at,
    interface_mu,
    total_magnetization,
    write_reports,
)
from interfaceforge.phase_diagram import (
    covered_subsystems,
    hull_report,
    missing_known_phases,
)
from interfaceforge.regime import BULK, FREE_SURFACE, measure_regime
from interfaceforge.separation_energy import EV_A2_TO_J_M2, _read_atoms, _structure_file

_OUTCAR = (
    " energy  without entropy=     {e:.6f}  energy(sigma->0) =     {e:.6f}\n"
    " General timing and accounting informations for this job\n"
)


def _run(directory: Path, species: list[tuple[str, int]], energy: float,
         box: tuple[float, float, float] = (10.0, 10.0, 20.0), vacuum: bool = False) -> Path:
    """A finished VASP run whose atoms fill the cell (no vacuum) unless asked."""

    directory.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, n in species)
    coords = []
    for i in range(total):  # quasirandom: spreads every axis, no degenerate gaps
        x, y = (i * 0.7548776662) % 1.0, (i * 0.5698402910) % 1.0
        z = ((i * 0.6180339887) % 1.0) * (0.35 if vacuum else 1.0)
        coords.append(f"  {x:.6f}  {y:.6f}  {z:.6f}")
    (directory / "POSCAR").write_text(
        "synthetic\n1.0\n"
        f"  {box[0]:.6f} 0.0 0.0\n  0.0 {box[1]:.6f} 0.0\n  0.0 0.0 {box[2]:.6f}\n"
        + " ".join(s for s, _ in species) + "\n"
        + " ".join(str(n) for _, n in species) + "\n"
        + "Direct\n" + "\n".join(coords) + "\n",
        encoding="utf-8",
    )
    (directory / "INCAR").write_text("IBRION = -1\nNSW = 0\n", encoding="utf-8")
    (directory / "OUTCAR").write_text(_OUTCAR.format(e=energy), encoding="utf-8")
    return directory


# Reference set chosen so the numbers are hand-checkable:
#   mu0(N)      = -16/2      = -8.0 eV
#   E(Ti)/atom  = -14/2      = -7.0 eV      E(Si)/atom = -10/2 = -5.0 eV
#   g(TiN)      = -18.5  =>  dHf = -18.5 + 7 + 8      = -3.5 eV/f.u.  -> bound -3.5
#   g(Si3N4)    = -55.0  =>  dHf = -55 + 15 + 32      = -8.0 eV/f.u.  -> bound -2.0
#   => dmu window [-2.0, 0], bound by Si3N4
def _phases(root: Path) -> dict[str, Path]:
    return {
        "TiN": _run(root / "TiN", [("Ti", 4), ("N", 4)], -74.0),
        "Si3N4": _run(root / "Si3N4", [("Si", 6), ("N", 8)], -110.0),
        "N2": _run(root / "N2", [("N", 2)], -16.0, box=(12.0, 12.0, 12.0)),
        "Ti": _run(root / "Ti", [("Ti", 2)], -14.0, box=(6.0, 6.0, 6.0)),
        "Si": _run(root / "Si", [("Si", 2)], -10.0, box=(6.0, 6.0, 6.0)),
    }


class RegimeTests(unittest.TestCase):
    def test_vacuum_slab_and_dense_cell_are_classified_apart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = _run(root / "dense", [("Ti", 8), ("N", 8)], -1.0)
            slab = _run(root / "slab", [("Ti", 8), ("N", 8)], -1.0, vacuum=True)
            self.assertEqual(measure_regime(_read_atoms(_structure_file(dense)))["regime"], BULK)
            self.assertEqual(
                measure_regime(_read_atoms(_structure_file(slab)))["regime"], FREE_SURFACE
            )


def _molecule(directory: Path, element: str, natoms: int, energy: float,
              moment: float | None) -> Path:
    """A molecular reference run whose OUTCAR reports (or omits) a total moment."""

    _run(directory, [(element, natoms)], energy, box=(12.0, 12.0, 12.0), vacuum=True)
    lines = [f"   free energy    TOTEN  =      {energy:.6f} eV"]
    for step in range(2):  # the audit must read the LAST SCF step, not the first
        value = 0.0 if moment is None else (moment * 0.5 if step == 0 else moment)
        lines.append(
            f"      number of electron     12.0000000 magnetization     {value:.7f}"
            if moment is not None
            else "      number of electron     12.0000000 magnetization"
        )
    lines.append(
        f" energy  without entropy=     {energy:.6f}  "
        f"energy(sigma->0) =     {energy:.6f}"
    )
    lines.append(" General timing and accounting informations for this job")
    (directory / "OUTCAR").write_text(
        chr(10).join(lines) + chr(10), encoding="utf-8"
    )
    return directory


class MolecularSpinTests(unittest.TestCase):
    def test_an_unpolarised_o2_reference_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o2 = _molecule(root / "O2", "O", 2, -9.9, None)
            with self.assertRaises(SafetyError) as caught:
                audit_molecular_spin("O2", o2, {"O": 2})
            message = str(caught.exception)
            self.assertIn("not spin-polarised at all", message)
            self.assertIn("~1 eV too high", message)
            self.assertIn("NUPDOWN=2", message)

    def test_a_triplet_o2_reference_passes_and_reads_the_last_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o2 = _molecule(root / "O2", "O", 2, -9.9, 2.0)
            self.assertAlmostEqual(total_magnetization(o2), 2.0)
            audit = audit_molecular_spin("O2", o2, {"O": 2})
            self.assertEqual(audit["status"], "PASS")
            self.assertAlmostEqual(audit["expected_moment_mub"], 2.0)

    def test_a_half_converged_triplet_is_flagged_without_refusing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o2 = _molecule(root / "O2", "O", 2, -9.9, 1.6)
            audit = audit_molecular_spin("O2", o2, {"O": 2})
            self.assertEqual(audit["status"], "CHECK")
            self.assertIn("only partly resolved", audit["note"])

    def test_an_unpolarised_n2_reference_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n2 = _molecule(root / "N2", "N", 2, -16.0, None)
            audit = audit_molecular_spin("N2", n2, {"N": 2})
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["expected_moment_mub"], 0.0)

    def test_a_magnetised_n2_reference_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            n2 = _molecule(root / "N2", "N", 2, -16.0, 1.0)
            audit = audit_molecular_spin("N2", n2, {"N": 2})
            self.assertEqual(audit["status"], "CHECK")
            self.assertIn("did not converge", audit["note"])

    def test_an_element_with_no_recorded_multiplicity_is_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            se = _molecule(root / "Se", "Se", 2, -7.0, None)
            audit = audit_molecular_spin("Se2", se, {"Se": 2})
            self.assertEqual(audit["status"], "NOT_CHECKED")
            self.assertIn("check the multiplicity yourself", audit["note"])

    def test_the_override_records_what_it_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            o2 = _molecule(root / "O2", "O", 2, -9.9, None)
            audit = audit_molecular_spin("O2", o2, {"O": 2}, strict=False)
            self.assertEqual(audit["status"], "OVERRIDDEN")
            self.assertIn("--allow-spin-mismatch", audit["note"])
            self.assertIn("~1 eV too high", audit["detail"])


class InterfaceMuTests(unittest.TestCase):
    def test_pairwise_window_is_bounded_by_the_binding_compound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            payload = interface_mu(
                [("balanced", iface)], phases=_phases(root), anion="N",
                n_interfaces=2, window_method="pairwise",
            )
            window = payload["chemical_potential_window"]
            self.assertEqual(payload["window_method"], "pairwise-formation-enthalpy")
            self.assertAlmostEqual(window["mu_anion_reference_ev"], -8.0)
            self.assertAlmostEqual(window["dmu_min_ev"], -2.0)
            self.assertEqual(window["dmu_max_ev"], 0.0)
            self.assertEqual(window["binding_compound"], "Si3N4")
            by_compound = {b["compound"]: b for b in window["bounds"]}
            self.assertAlmostEqual(by_compound["TiN"]["formation_enthalpy_ev_per_fu"], -3.5)
            self.assertAlmostEqual(by_compound["Si3N4"]["formation_enthalpy_ev_per_fu"], -8.0)

    def test_hull_window_agrees_with_the_pairwise_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            hull = interface_mu(
                [("balanced", iface)], phases=phases, anion="N", n_interfaces=2,
                window_method="hull",
            )["chemical_potential_window"]
            pairwise = interface_mu(
                [("balanced", iface)], phases=phases, anion="N", n_interfaces=2,
                window_method="pairwise",
            )["chemical_potential_window"]
            self.assertEqual(hull["method"], "convex-hull")
            # the hull is the rigorous route; on a system with no competing
            # ternary it must reproduce the pairwise formation-enthalpy bound
            self.assertAlmostEqual(hull["dmu_min_ev"], pairwise["dmu_min_ev"], places=6)
            self.assertAlmostEqual(hull["dmu_max_ev"], pairwise["dmu_max_ev"], places=6)
            self.assertEqual(hull["binding_compound"], pairwise["binding_compound"])
            self.assertEqual(hull["hull"]["unstable_phases"], [])
            self.assertEqual(hull["competing_stable_phases"], [])

    def test_hull_names_a_competing_silicide(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            phases["TiSi2"] = _run(root / "TiSi2", [("Ti", 2), ("Si", 4)], -46.0)
            window = interface_mu(
                [("balanced", iface)], phases=phases, anion="N", n_interfaces=2,
            )["chemical_potential_window"]
            self.assertEqual(window["competing_stable_phases"], ["TiSi2"])

    def test_hull_names_the_known_phases_it_was_not_given(self) -> None:
        """An incomplete hull says so: its window is an upper limit, not the answer."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            window = interface_mu(
                [("balanced", iface)], phases=_phases(root), anion="N", n_interfaces=2,
            )["chemical_potential_window"]
            # every Ti-N and Ti-Si phase of the system is absent from _phases()
            self.assertIn("Ti2N (mp-8282)", window["missing_known_phases"])
            self.assertIn("Ti5Si3 (mp-2108)", window["missing_known_phases"])
            self.assertIn("TiSi2 (mp-2582)", window["missing_known_phases"])
            self.assertIn("too wide", window["hull"]["completeness_note"])

    def test_missing_known_phases_ignores_other_subsystems_and_polymorphs(self) -> None:
        with tempfile.TemporaryDirectory():
            # a Ti-N hull must not be scolded for lacking silicides
            ti_n = [row["formula"] for row in missing_known_phases(["Ti", "N"], ["TiN", "Ti", "N2"])]
            self.assertEqual(ti_n, ["Ti2N"])
            # a polymorph counts as coverage of its composition: supplying only
            # anatase must silence the rutile entry, not report TiO2 as missing
            ti_o = [row["formula"] for row in missing_known_phases(
                ["Ti", "O"], ["TiO2", "TiO", "Ti2O3", "Ti3O5", "Ti4O7", "Ti", "O2"]
            )]
            self.assertEqual(ti_o, [])
            self.assertEqual(
                [row["formula"] for row in missing_known_phases(
                    ["Ti", "O"], ["TiO2", "TiO", "Ti2O3", "Ti", "O2"]
                )],
                ["Ti3O5", "Ti4O7"],  # the Magneli phases of the reduced branch
            )

    def test_a_complete_hull_says_nothing_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            small = (6.0, 6.0, 6.0)
            # every Ti-N/Ti-Si phase supplied, priced above the hull so the window
            # itself is untouched -- only the completeness claim changes. Ti2N is a
            # competing nitride, not a constituent, so it goes in --aux-phase.
            aux = {"Ti2N": _run(root / "Ti2N", [("Ti", 4), ("N", 2)], -30.0, box=small)}
            phases["Ti5Si3"] = _run(root / "Ti5Si3", [("Ti", 5), ("Si", 3)], -50.0, box=small)
            phases["Ti5Si4"] = _run(root / "Ti5Si4", [("Ti", 5), ("Si", 4)], -55.0, box=small)
            phases["Ti3Si"] = _run(root / "Ti3Si", [("Ti", 3), ("Si", 1)], -26.0, box=small)
            phases["TiSi"] = _run(root / "TiSi", [("Ti", 2), ("Si", 2)], -24.0, box=small)
            phases["TiSi2"] = _run(root / "TiSi2", [("Ti", 2), ("Si", 4)], -34.0, box=small)
            window = interface_mu(
                [("balanced", iface)], phases=phases, auxiliary_phases=aux,
                anion="N", n_interfaces=2,
            )["chemical_potential_window"]
            self.assertEqual(window["missing_known_phases"], [])
            self.assertAlmostEqual(window["dmu_min_ev"], -2.0)
            self.assertIn("every phase", window["hull"]["completeness_note"])

    def test_a_competing_nitride_cuts_the_window_without_being_a_constituent(self) -> None:
        """Ti2N is the phase TiN decomposes toward, not a phase the interface is made of."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            # E(Ti2N) = -28 eV/f.u. puts it below the Ti-TiN tie line, so the
            # N-poor limit of TiN becomes 2 TiN -> Ti2N + N at dmu_N = -1.0 eV,
            # tighter than the -2.0 eV that Si3N4 -> 3 Si + 2 N2 alone allows
            aux = {"Ti2N": _run(root / "Ti2N", [("Ti", 4), ("N", 2)], -56.0, box=(6.0, 6.0, 6.0))}
            payload = interface_mu(
                [("balanced", iface)], phases=phases, auxiliary_phases=aux,
                anion="N", n_interfaces=2,
            )
            window = payload["chemical_potential_window"]
            self.assertAlmostEqual(window["dmu_min_ev"], -1.0, places=6)
            self.assertEqual(window["binding_compound"], "TiN")
            self.assertIn("Ti2N", window["competing_stable_phases"])
            self.assertNotIn("Ti2N (mp-8282)", window["missing_known_phases"])
            # and it stays out of the decomposition: the cell is still 4 TiN + 1 Si3N4
            units = payload["interfaces"][0]["decomposition"]["formula_units"]
            self.assertEqual(sorted(units), ["Si3N4", "TiN"])

    def test_a_system_outside_the_built_in_list_says_it_was_not_checked(self) -> None:
        """Silence from the completeness check must not read as a clean bill of health."""

        self.assertEqual(covered_subsystems({"Ni", "O", "H"}), [])
        report = hull_report(
            {
                "NiO": {"composition": {"Ni": 2, "O": 2}, "energy_ev": -20.0},
                "Ni": {"composition": {"Ni": 2}, "energy_ev": -10.0},
                "O2": {"composition": {"O": 2}, "energy_ev": -9.0},
            },
            ["NiO"],
            "O",
        )
        self.assertEqual(report["missing_known_phases"], [])
        self.assertIn("completeness NOT checked", report["completeness_note"])

    def test_the_report_warns_that_an_incomplete_hull_gives_an_upper_limit(self) -> None:
        """The markdown is what gets read; the warning cannot live only in the JSON."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            payload = interface_mu(
                [("balanced", iface)], phases=_phases(root), anion="N", n_interfaces=2,
            )
            write_reports(payload, root / "out")
            report = (root / "out" / "interface_mu.md").read_text(encoding="utf-8")
            self.assertIn("upper limit, not the window", report)
            self.assertIn("Ti5Si3 (mp-2108)", report)
            self.assertTrue(report.isascii(), "the generated report must stay ASCII")

    def test_the_report_states_the_anion_reference_spin_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            payload = interface_mu(
                [("balanced", iface)], phases=_phases(root), anion="N", n_interfaces=2,
            )
            self.assertEqual(payload["reference_phases"]["N2"]["spin"]["status"], "PASS")
            self.assertIsNone(payload["reference_phases"]["TiN"]["spin"])
            write_reports(payload, root / "out")
            report = (root / "out" / "interface_mu.md").read_text(encoding="utf-8")
            self.assertIn("Anion reference N2: not spin-polarised (expected 0.0 muB)", report)

    def test_two_compounds_sharing_a_cation_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            phases["Ti2N"] = _run(root / "Ti2N", [("Ti", 4), ("N", 2)], -56.0, box=(6.0, 6.0, 6.0))
            with self.assertRaises(SafetyError) as caught:
                interface_mu([("balanced", iface)], phases=phases, anion="N", n_interfaces=2)
            self.assertIn("share the cation Ti", str(caught.exception))
            self.assertIn("--aux-phase", str(caught.exception))

    def test_a_phase_cannot_be_both_a_constituent_and_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            with self.assertRaises(SafetyError) as caught:
                interface_mu(
                    [("balanced", iface)], phases=phases,
                    auxiliary_phases={"TiN": phases["TiN"]}, anion="N", n_interfaces=2,
                )
            self.assertIn("both --phase and --aux-phase", str(caught.exception))

    def test_the_anion_reference_cannot_be_demoted_to_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            anion_reference = phases.pop("N2")
            with self.assertRaises(SafetyError) as caught:
                interface_mu(
                    [("balanced", iface)], phases=phases,
                    auxiliary_phases={"N2": anion_reference}, anion="N", n_interfaces=2,
                )
            self.assertIn("zero of the chemical-potential scale", str(caught.exception))

    def test_two_elemental_references_for_one_element_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            phases = _phases(root)
            # omega-Ti alongside hcp Ti: silently overwriting one would move
            # every formation enthalpy that the N-poor bound is built from
            phases["Ti_omega"] = _run(root / "Ti_w", [("Ti", 2)], -13.5, box=(6.0, 6.0, 6.0))
            with self.assertRaises(SafetyError) as caught:
                interface_mu([("balanced", iface)], phases=phases, anion="N", n_interfaces=2)
            self.assertIn("two elemental Ti references", str(caught.exception))

    def test_balanced_cell_is_chemical_potential_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Ti4 Si3 N8 = 4 TiN + 1 Si3N4 exactly; E_int - reference = -128 + 129 = +1 eV
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            payload = interface_mu(
                [("balanced", iface)], phases=_phases(root), anion="N", n_interfaces=2
            )
            row = payload["interfaces"][0]
            self.assertEqual(row["regime"]["regime"], BULK)
            self.assertTrue(row["decomposition"]["stoichiometric"])
            self.assertEqual(row["decomposition"]["formula_units"], {"TiN": 4.0, "Si3N4": 1.0})
            self.assertAlmostEqual(row["decomposition"]["anion_excess"], 0.0)
            self.assertAlmostEqual(row["dft"]["slope_j_per_m2_per_ev"], 0.0)
            self.assertAlmostEqual(row["dft"]["gamma0_j_per_m2"], 1.0 / 200.0 * EV_A2_TO_J_M2)
            self.assertAlmostEqual(
                row["dft"]["gamma_anion_poor_j_per_m2"], row["dft"]["gamma_anion_rich_j_per_m2"]
            )
            self.assertEqual(payload["chemical_potential_dependent"], [])

    def test_excess_anion_makes_gamma_a_line_with_that_slope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Ti4 Si3 N9: one N beyond 4 TiN + 1 Si3N4  =>  dn = +1
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 9)], -136.0)
            payload = interface_mu(
                [("n-rich", iface)], phases=_phases(root), anion="N", n_interfaces=2
            )
            row = payload["interfaces"][0]
            self.assertFalse(row["decomposition"]["stoichiometric"])
            self.assertAlmostEqual(row["decomposition"]["anion_excess"], 1.0)
            # gamma0 = (-136 + 129 + 8) / 200 * conv ; slope = -1/200 * conv
            self.assertAlmostEqual(row["dft"]["gamma0_j_per_m2"], 1.0 / 200.0 * EV_A2_TO_J_M2)
            self.assertAlmostEqual(
                row["dft"]["slope_j_per_m2_per_ev"], -1.0 / 200.0 * EV_A2_TO_J_M2
            )
            self.assertAlmostEqual(
                row["dft"]["gamma_anion_poor_j_per_m2"], 3.0 / 200.0 * EV_A2_TO_J_M2
            )
            self.assertEqual(payload["chemical_potential_dependent"], ["n-rich"])
            self.assertAlmostEqual(gamma_at(row["dft"], -1.0), 2.0 / 200.0 * EV_A2_TO_J_M2)

    def test_a_vacuum_slab_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slab = _run(root / "slab", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0, vacuum=True)
            with self.assertRaisesRegex(SafetyError, "free-surface structure"):
                interface_mu([("slab", slab)], phases=_phases(root), anion="N")

    def test_mlip_committee_shifts_the_intercept_not_the_slope(self) -> None:
        def fake_mace(models, atoms_by_key, device):
            # every member: interface 0.4 eV higher, references exact
            return {
                f"seed_{k}": {
                    key: (-136.0 + 0.4 + 0.02 * k) if key.startswith("iface::")
                    else {"phase::TiN": -74.0, "phase::Si3N4": -110.0}[key]
                    for key in atoms_by_key
                }
                for k in range(4)
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 9)], -136.0)
            with patch("interfaceforge.interface_mu._mace_energies", fake_mace):
                payload = interface_mu(
                    [("n-rich", iface)], phases=_phases(root), anion="N",
                    n_interfaces=2, mace_models=["a", "b", "c", "d"],
                )
            block = payload["interfaces"][0]["mlip"]["mace"]
            self.assertEqual(block["members"], 4)
            # slope is structural (composition + area), so it must match DFT exactly
            self.assertAlmostEqual(
                block["slope_j_per_m2_per_ev"],
                payload["interfaces"][0]["dft"]["slope_j_per_m2_per_ev"],
            )
            # +0.43 eV mean on the interface only -> that / 200 * conv
            self.assertAlmostEqual(
                block["delta_vs_dft_j_per_m2"], 0.43 / 200.0 * EV_A2_TO_J_M2, places=6
            )
            self.assertGreater(block["committee_spread_j_per_m2"], 0.0)

    def test_mlip_is_never_evaluated_on_the_molecular_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iface = _run(root / "iface", [("Ti", 4), ("Si", 3), ("N", 8)], -128.0)
            seen: dict[str, list[str]] = {}

            def fake_mace(models, atoms_by_key, device):
                seen["keys"] = sorted(atoms_by_key)
                return {
                    "m0": {
                        key: {"iface::balanced": -128.0, "phase::TiN": -74.0,
                              "phase::Si3N4": -110.0}[key]
                        for key in atoms_by_key
                    }
                }

            with patch("interfaceforge.interface_mu._mace_energies", fake_mace):
                payload = interface_mu(
                    [("balanced", iface)], phases=_phases(root), anion="N",
                    n_interfaces=2, mace_models=["a"],
                )
            self.assertEqual(seen["keys"], ["iface::balanced", "phase::Si3N4", "phase::TiN"])
            self.assertNotIn("phase::N2", seen["keys"])
            self.assertNotIn("phase::Ti", seen["keys"])
            self.assertEqual(payload["regime"], BULK)


if __name__ == "__main__":
    unittest.main()
