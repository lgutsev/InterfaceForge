from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interfaceforge.chempot_map import (
    map_report,
    place_conditions,
    plane_constraints,
    stability_polygon,
    write_reports,
)
from interfaceforge.errors import SafetyError
from interfaceforge.phase_diagram import build_hull, hull_report

#: A Ti-N-O system whose whole 2D region can be written down in one line.
#:
#: Every elemental reference is set to zero energy, so a compound's total energy
#: IS its formation energy and the arithmetic is visible:
#:     dHf(TiN) = -4 eV/f.u.,  dHf(TiO2) = -10 eV/f.u.,  dHf(TiO) = -1 eV/f.u.
#:
#: Holding TiN in equilibrium gives dmu_Ti = -4 - dmu_N, and substituting that
#: into the TiO2 condition dmu_Ti + 2 dmu_O <= -10 leaves
#:
#:     -dmu_N + 2 dmu_O <= -6        i.e.   dmu_O <= (dmu_N - 6) / 2
#:
#: with dmu_N in [-4, 0] (the upper end from N2, the lower from Ti not
#: precipitating). The corners are therefore exactly (0, -3) and (-4, -5).
#: TiO is far too weak to bind and must come out inactive.
_TERNARY = {
    "TiN":  {"composition": {"Ti": 1, "N": 1}, "energy_ev": -4.0},
    "TiO2": {"composition": {"Ti": 1, "O": 2}, "energy_ev": -10.0},
    "TiO":  {"composition": {"Ti": 1, "O": 1}, "energy_ev": -1.0},
    "Ti":   {"composition": {"Ti": 1},         "energy_ev": 0.0},
    "N2":   {"composition": {"N": 2},          "energy_ev": 0.0},
    "O2":   {"composition": {"O": 2},          "energy_ev": 0.0},
}

#: The same toy Ti-Si-N-O set the 0 K tests use, so the 2D region and the 1D
#: window are checked against each other on identical inputs.
_QUATERNARY = {
    "TiN":   {"composition": {"Ti": 4, "N": 4}, "energy_ev": -74.0},
    "Si3N4": {"composition": {"Si": 6, "N": 8}, "energy_ev": -110.0},
    "TiO2":  {"composition": {"Ti": 2, "O": 4}, "energy_ev": -60.0},
    "SiO2":  {"composition": {"Si": 3, "O": 6}, "energy_ev": -80.0},
    "N2":    {"composition": {"N": 2},          "energy_ev": -16.0},
    "O2":    {"composition": {"O": 2},          "energy_ev": -9.0},
    "Ti":    {"composition": {"Ti": 2},         "energy_ev": -14.0},
    "Si":    {"composition": {"Si": 2},         "energy_ev": -10.0},
}


def _polygon(phases, compounds, axes=("N", "O"), **kwargs):
    return stability_polygon(build_hull(phases), compounds, axes, **kwargs)


class AnalyticRegionTest(unittest.TestCase):
    """A system small enough that the answer is checkable by hand."""

    def test_the_corners_are_where_the_algebra_says(self):
        region = _polygon(_TERNARY, ["TiN"])
        corners = sorted(tuple(round(c, 9) for c in v["point"]) for v in region["vertices"])
        self.assertEqual(corners, [(-4.0, -5.0), (0.0, -3.0)])

    def test_the_binding_line_has_the_derived_slope_and_intercept(self):
        """dmu_O <= (dmu_N - 6)/2: slope 1/2, intercept -3."""

        region = _polygon(_TERNARY, ["TiN"])
        row = next(r for r in region["constraints"] if r["phase"] == "TiO2")
        a, b = row["coefficients"]
        self.assertAlmostEqual(-a / b, 0.5, places=12)          # slope
        self.assertAlmostEqual(row["bound_ev"] / b, -3.0, places=12)  # intercept

    def test_a_phase_that_never_binds_is_reported_inactive(self):
        region = _polygon(_TERNARY, ["TiN"])
        self.assertIn("TiO2", region["active_phases"])
        self.assertIn("TiO", region["inactive_phases"])
        self.assertIn("O2", region["inactive_phases"])
        row = next(r for r in region["constraints"] if r["phase"] == "TiO")
        self.assertFalse(row["active"])
        # and the report says by how much it misses, rather than just "no"
        self.assertGreater(row["slack_at_best_ev"], 0.0)

    def test_an_above_hull_phase_can_never_bind(self):
        """A phase off the hull is unstable at every reservoir, so its line is slack."""

        hull = build_hull(_TERNARY)
        above = next(r for r in hull["phases"] if r["phase"] == "TiO")
        self.assertGreater(above["e_above_hull_ev_per_atom"], 0.0)
        self.assertIn("TiO", _polygon(_TERNARY, ["TiN"])["inactive_phases"])

    def test_the_edges_name_the_phase_that_controls_them(self):
        edges = {e["edge"]: e for e in _polygon(_TERNARY, ["TiN"])["edges"]}
        self.assertAlmostEqual(edges["N-rich"]["value_ev"], 0.0, places=9)
        self.assertEqual(edges["N-rich"]["controlled_by"], ["N2"])
        self.assertAlmostEqual(edges["N-poor"]["value_ev"], -4.0, places=9)
        self.assertEqual(edges["N-poor"]["controlled_by"], ["Ti"])
        self.assertAlmostEqual(edges["O-rich"]["value_ev"], -3.0, places=9)
        self.assertEqual(edges["O-rich"]["controlled_by"], ["N2", "TiO2"])
        self.assertIsNone(edges["O-poor"]["value_ev"])

    def test_an_oxygen_poor_region_is_unbounded_and_says_so(self):
        """Nothing forces oxygen in, so dmu_O runs to -infinity. Do not pretend otherwise."""

        region = _polygon(_TERNARY, ["TiN"])
        self.assertFalse(region["bounded"])
        self.assertIsNone(region["extent_ev"]["O"]["min"])
        self.assertIn("UNBOUNDED", region["note"])
        self.assertIn("clip", region["note"])

    def test_the_eliminated_potential_is_reported_as_a_function_of_the_axes(self):
        region = _polygon(_TERNARY, ["TiN"])
        reduction = region["reduction"]
        self.assertEqual(reduction["eliminated"], ["Ti"])
        self.assertAlmostEqual(reduction["offset_ev"]["Ti"], -4.0, places=12)
        index = ["N", "O", "Ti"].index("Ti")
        self.assertAlmostEqual(reduction["gradient"][index][0], -1.0, places=12)
        self.assertAlmostEqual(reduction["gradient"][index][1], 0.0, places=12)

    def test_the_constraints_are_scale_free(self):
        """Doubling every cell must not move a boundary."""

        doubled = {
            name: {
                "composition": {el: 2 * n for el, n in phase["composition"].items()},
                "energy_ev": 2 * phase["energy_ev"],
            }
            for name, phase in _TERNARY.items()
        }
        a = _polygon(_TERNARY, ["TiN"])["vertices"]
        b = _polygon(doubled, ["TiN"])["vertices"]
        for left, right in zip(a, b, strict=True):
            self.assertAlmostEqual(left["point"][0], right["point"][0], places=9)
            self.assertAlmostEqual(left["point"][1], right["point"][1], places=9)


class QuaternaryRegionTest(unittest.TestCase):
    """Ti-Si-N-O: two constituents, two free axes, a kinked upper envelope."""

    def test_the_corners_match_the_hand_derivation(self):
        region = _polygon(_QUATERNARY, ["TiN", "Si3N4"])
        corners = sorted(tuple(round(c, 6) for c in v["point"]) for v in region["vertices"])
        self.assertEqual(
            corners, [(-2.0, -6.333333), (-1.5, -6.0), (0.0, -5.25)]
        )

    def test_the_two_oxide_boundaries_have_the_documented_slopes(self):
        """docs/interface-mu.md already states these are 1/2 (TiO2) and 2/3 (SiO2)."""

        region = _polygon(_QUATERNARY, ["TiN", "Si3N4"])
        slopes = {}
        for name in ("TiO2", "SiO2"):
            a, b = next(r for r in region["constraints"] if r["phase"] == name)["coefficients"]
            slopes[name] = -a / b
        self.assertAlmostEqual(slopes["TiO2"], 0.5, places=12)
        self.assertAlmostEqual(slopes["SiO2"], 2.0 / 3.0, places=12)

    def test_which_oxide_binds_changes_along_the_nitrogen_axis(self):
        """The kink at dmu_N = -1.5 is where TiO2 hands over to SiO2."""

        region = _polygon(_QUATERNARY, ["TiN", "Si3N4"])
        kink = next(
            v for v in region["vertices"] if set(v["phases"]) == {"TiO2", "SiO2"}
        )
        self.assertAlmostEqual(kink["point"][0], -1.5, places=9)
        self.assertAlmostEqual(kink["point"][1], -6.0, places=9)

    def test_the_nitrogen_extent_reproduces_the_existing_1d_window(self):
        """The 1D window is the projection of this region. They must agree.

        This is the cross-check that would have caught the inconsistency that
        motivated the module: one number computed two independent ways.
        """

        region = _polygon(_QUATERNARY, ["TiN", "Si3N4"])
        window = hull_report(_QUATERNARY, ["TiN", "Si3N4"], "N")[
            "chemical_potential_window"
        ]
        self.assertAlmostEqual(
            region["extent_ev"]["N"]["min"], window["dmu_min_ev"], places=7
        )
        self.assertAlmostEqual(
            region["extent_ev"]["N"]["max"], window["dmu_max_ev"], places=7
        )

    def test_the_oxygen_extent_is_exact_where_the_1d_bisection_only_approximates(self):
        """The joint LP gives -5.25 exactly and names the phase that generates it.

        The 1D open-element path bisects to a tolerance and attributes the bound
        to the compound that fell over, not to the phase whose stability
        condition draws the line. Both are defensible answers to different
        questions; this pins that the 2D route is the exact, correctly
        attributed one.
        """

        region = _polygon(_QUATERNARY, ["TiN", "Si3N4"])
        self.assertAlmostEqual(region["extent_ev"]["O"]["max"], -5.25, places=9)
        o_rich = next(e for e in region["edges"] if e["edge"] == "O-rich")
        self.assertIn("TiO2", o_rich["controlled_by"])

        limit = hull_report(_QUATERNARY, ["TiN", "Si3N4"], "O")[
            "chemical_potential_window"
        ]
        self.assertEqual(limit["method"], "grand-potential-open-element")
        self.assertAlmostEqual(limit["dmu_max_ev"], -5.25, places=3)
        self.assertNotAlmostEqual(limit["dmu_max_ev"], -5.25, places=6)


class PlaneDefinitionTest(unittest.TestCase):
    def test_too_few_constituents_is_refused_with_the_count(self):
        with self.assertRaises(SafetyError) as caught:
            _polygon(_QUATERNARY, ["TiN"])
        message = str(caught.exception)
        self.assertIn("degree(s) of freedom", message)
        self.assertIn("exactly 2", message)

    def test_too_many_constituents_is_refused(self):
        with self.assertRaises(SafetyError) as caught:
            _polygon(_QUATERNARY, ["TiN", "Si3N4", "TiO2"])
        self.assertIn("over-determined", str(caught.exception))

    def test_a_fixed_potential_can_supply_the_missing_equality(self):
        region = _polygon(_QUATERNARY, ["TiN"], fixed_dmu={"Si": -1.0})
        self.assertEqual(region["fixed_dmu_ev"], {"Si": -1.0})
        self.assertEqual(region["reduction"]["degrees_of_freedom"], 2)

    def test_an_axis_cannot_also_be_fixed(self):
        with self.assertRaises(SafetyError) as caught:
            _polygon(_QUATERNARY, ["TiN"], fixed_dmu={"O": -1.0})
        self.assertIn("not a free axis", str(caught.exception))

    def test_an_axis_without_an_elemental_reference_is_refused(self):
        with self.assertRaises(SafetyError) as caught:
            _polygon(_TERNARY, ["TiN"], axes=("N", "Xe"))
        self.assertIn("not in the hull", str(caught.exception))

    def test_two_identical_axes_are_refused(self):
        with self.assertRaises(SafetyError):
            _polygon(_TERNARY, ["TiN"], axes=("N", "N"))

    def test_an_unstable_constituent_cannot_define_the_plane(self):
        with self.assertRaises(SafetyError) as caught:
            _polygon(_TERNARY, ["TiO"])
        self.assertIn("above the convex hull", str(caught.exception))

    def test_axes_the_equalities_already_tie_together_are_refused(self):
        """If dmu_O is determined by dmu_N there is no plane, only a line."""

        # Holding TiN and TiO2 both in equilibrium fixes dmu_O in terms of dmu_N.
        with self.assertRaises(SafetyError) as caught:
            plane_constraints(build_hull(_TERNARY), ["TiN", "TiO2"], ("N", "O"))
        message = str(caught.exception)
        self.assertTrue(
            "not independent coordinates" in message or "degree(s) of freedom" in message,
            message,
        )


class ConditionPlacementTest(unittest.TestCase):
    """Putting a real atmosphere on the diagram."""

    def _conditions(self, **kwargs):
        return map_report(
            _TERNARY, ["TiN"], ("N", "O"),
            conditions=[
                {"label": "hot dry", "temperature_k": 600.0,
                 "pressures": {"N": 1.0, "O": 1.0e-20}, "pressure_unit": "atm"},
                {"label": "air", "temperature_k": 300.0,
                 "pressures": {"N": 0.78, "O": 0.21}, "pressure_unit": "atm"},
            ],
            **kwargs,
        )["conditions"]

    def test_a_condition_carries_its_own_thermochemistry(self):
        hot, air = self._conditions()
        self.assertEqual(hot["temperature_k"], 600.0)
        self.assertLess(hot["dmu_ev_per_atom"]["O"], air["dmu_ev_per_atom"]["O"])
        self.assertIn("N", hot["reservoirs"])
        self.assertIn("thermal_correction_atomic_ev", hot["reservoirs"]["O"])

    def test_an_atmosphere_outside_the_region_names_what_undercuts_it(self):
        """The O-rich bound here is -3 eV; ordinary air is nowhere near it."""

        _, air = self._conditions()
        self.assertFalse(air["inside"])
        self.assertEqual([v["phase"] for v in air["violated_phases"]], ["TiO2"])
        self.assertGreater(air["violated_phases"][0]["violation_ev"], 0.0)
        self.assertIn("TiO2", air["interpretation"])

    def test_the_reported_point_is_the_pair_of_atomic_deltas(self):
        hot, _ = self._conditions()
        self.assertAlmostEqual(hot["point"][0], hot["dmu_ev_per_atom"]["N"], places=15)
        self.assertAlmostEqual(hot["point"][1], hot["dmu_ev_per_atom"]["O"], places=15)

    def test_an_outside_condition_reports_the_pressure_it_would_need(self):
        """"Outside by 1.6 eV" is hard to act on; a required p_O2 is not."""

        _, air = self._conditions()
        limit = air["limiting_pressure"]
        self.assertTrue(limit["feasible"])
        self.assertEqual(limit["axis"], "O")
        # the ceiling is the O-rich boundary evaluated at this condition's dmu_N
        self.assertLess(limit["dmu_ceiling_ev"], air["dmu_ev_per_atom"]["O"])
        self.assertLess(limit["log10_pressure_bar"], 0.0)
        self.assertIn("kinetic", limit["note"])

    def test_the_required_pressure_lands_exactly_on_the_boundary(self):
        """Feed the reported pressure back in and the condition must sit on the edge."""

        region = _polygon(_TERNARY, ["TiN"])
        first = place_conditions(
            region,
            [{"temperature_k": 600.0, "pressures": {"N": 1.0, "O": 1.0e-10}}],
        )[0]
        needed_bar = first["limiting_pressure"]["pressure_bar"]
        self.assertGreater(needed_bar, 0.0)
        again = place_conditions(
            region,
            [{"temperature_k": 600.0, "pressures": {"N": 1.0, "O": needed_bar},
              "pressure_unit": "bar"}],
        )[0]
        self.assertAlmostEqual(
            again["dmu_ev_per_atom"]["O"],
            first["limiting_pressure"]["dmu_ceiling_ev"],
            places=9,
        )
        # on the boundary means inside to within tolerance
        self.assertAlmostEqual(again["closest_boundary_ev"], 0.0, places=9)

    def test_an_unreachably_small_pressure_is_reported_not_overflowed(self):
        """exp() underflows long before the answer stops meaning 'unreachable'."""

        # A wildly stable oxide pushes the required pressure past any float.
        phases = dict(_TERNARY)
        phases["TiO2"] = {"composition": {"Ti": 1, "O": 2}, "energy_ev": -400.0}
        region = _polygon(phases, ["TiN"])
        row = place_conditions(
            region, [{"temperature_k": 300.0, "pressures": {"N": 1.0, "O": 1.0e-20}}]
        )[0]
        limit = row["limiting_pressure"]
        self.assertEqual(limit["pressure_bar"], 0.0)
        self.assertIsNone(limit["log10_pressure_bar"])
        self.assertIn("representable as a float", limit["note"])

    def test_a_condition_missing_an_axis_pressure_is_refused(self):
        region = _polygon(_TERNARY, ["TiN"])
        with self.assertRaises(SafetyError) as caught:
            place_conditions(region, [{"temperature_k": 600.0, "pressures": {"N": 1.0}}])
        self.assertIn("partial pressure for every axis", str(caught.exception))

    def test_a_nonphysical_condition_is_refused(self):
        region = _polygon(_TERNARY, ["TiN"])
        nonphysical = [
            ("temperature at absolute zero", -5.0, {"N": 1.0, "O": 1.0e-10}),
            ("zero temperature", 0.0, {"N": 1.0, "O": 1.0e-10}),
            ("zero partial pressure", 600.0, {"N": 1.0, "O": 0.0}),
            ("negative partial pressure", 600.0, {"N": -1.0, "O": 1.0e-10}),
        ]
        for label, temperature, pressures in nonphysical:
            with self.subTest(case=label), self.assertRaises(SafetyError):
                place_conditions(
                    region,
                    [{"temperature_k": temperature, "pressures": pressures}],
                )


class ZeroKelvinPreservationTest(unittest.TestCase):
    """The 0 K workflow must be exactly what it was before this layer existed."""

    def test_the_1d_nitrogen_window_is_unchanged(self):
        window = hull_report(_QUATERNARY, ["TiN", "Si3N4"], "N")[
            "chemical_potential_window"
        ]
        self.assertAlmostEqual(window["dmu_min_ev"], -2.0, places=9)
        self.assertEqual(window["dmu_max_ev"], 0.0)
        self.assertEqual(window["dmu_min_set_by"], "Si")
        self.assertEqual(window["dmu_max_set_by"], "N2")
        self.assertEqual(window["method"], "convex-hull")
        self.assertEqual(window["solver"], "joint-coexistence-linear-program")
        self.assertAlmostEqual(window["mu_anion_reference_ev"], -8.0, places=12)

    def test_the_1d_oxygen_limit_is_unchanged(self):
        window = hull_report(_QUATERNARY, ["TiN", "Si3N4"], "O")[
            "chemical_potential_window"
        ]
        self.assertEqual(window["method"], "grand-potential-open-element")
        self.assertAlmostEqual(window["dmu_max_ev"], -5.25, places=4)
        self.assertEqual(window["dmu_max_set_by"], "TiN")
        self.assertIsNone(window["dmu_min_ev"])
        self.assertAlmostEqual(window["mu_anion_reference_ev"], -4.5, places=12)

    def test_the_0k_payload_keys_are_unchanged(self):
        """A new layer must add commands, not reshape the existing payload."""

        payload = hull_report(_QUATERNARY, ["TiN", "Si3N4"], "N")
        self.assertEqual(
            sorted(payload),
            [
                "chemical_potential_window", "completeness_note", "elements",
                "hull_note", "missing_known_phases", "phases", "schema_version",
                "stable_phases", "unstable_phases",
            ],
        )
        self.assertEqual(payload["schema_version"], 1)

    def test_no_temperature_or_pressure_leaks_into_the_0k_payload(self):
        blob = json.dumps(hull_report(_QUATERNARY, ["TiN", "Si3N4"], "N")).lower()
        for leaked in ("temperature", "pressure", "shomate", "janaf", "kelvin"):
            self.assertNotIn(leaked, blob)

    def test_the_2d_map_without_conditions_is_pure_0k(self):
        payload = map_report(_QUATERNARY, ["TiN", "Si3N4"], ("N", "O"))
        self.assertEqual(payload["conditions"], [])
        blob = json.dumps(payload["stability_region"]).lower()
        self.assertNotIn("janaf", blob)
        self.assertNotIn("shomate", blob)


class ReportTest(unittest.TestCase):
    def test_the_payload_round_trips_through_json(self):
        payload = map_report(
            _TERNARY, ["TiN"], ("N", "O"),
            conditions=[{"label": "x", "temperature_k": 600.0,
                         "pressures": {"N": 1.0, "O": 1.0e-15}, "pressure_unit": "atm"}],
        )
        restored = json.loads(json.dumps(payload, default=str))
        self.assertEqual(
            restored["stability_region"]["active_phases"],
            payload["stability_region"]["active_phases"],
        )

    def test_the_plot_data_is_deterministic(self):
        first = map_report(_TERNARY, ["TiN"], ("N", "O"))["stability_region"]
        second = map_report(_TERNARY, ["TiN"], ("N", "O"))["stability_region"]
        self.assertEqual(first["polygon_ev"], second["polygon_ev"])
        self.assertEqual(first["vertices"], second["vertices"])
        self.assertEqual(first["clip_box_ev"], second["clip_box_ev"])

    def test_an_explicit_clip_is_honoured_and_recorded(self):
        payload = map_report(
            _TERNARY, ["TiN"], ("N", "O"), clip={"N": (-5.0, 0.5), "O": (-8.0, 0.0)}
        )
        region = payload["stability_region"]
        self.assertEqual(region["clip_box_ev"], {"N": [-5.0, 0.5], "O": [-8.0, 0.0]})
        # the clip is a drawing window, not a constraint: the region is still unbounded
        self.assertFalse(region["bounded"])
        for point in region["polygon_ev"]:
            self.assertGreaterEqual(point[0], -5.0 - 1.0e-9)
            self.assertLessEqual(point[1], 0.0 + 1.0e-9)

    def test_the_written_reports_carry_the_inequalities_and_provenance(self):
        payload = map_report(
            _TERNARY, ["TiN"], ("N", "O"),
            conditions=[{"label": "demo", "temperature_k": 600.0,
                         "pressures": {"N": 1.0, "O": 1.0e-15}, "pressure_unit": "atm"}],
        )
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            outputs = write_reports(payload, out)
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["constraints_csv"]).exists())
            self.assertTrue(Path(outputs["conditions_csv"]).exists())
            markdown = (out / "chempot_map.md").read_text(encoding="utf-8")
            self.assertIn("dmu_O <=", markdown.replace("dmu(O) <=", "dmu_O <="))
            self.assertIn("Active", markdown)
            self.assertIn("TiO2", markdown)
            csv_text = (out / "chempot_map_constraints.csv").read_text(encoding="utf-8")
            self.assertIn("inequality", csv_text)
            self.assertIn("TiO", csv_text)

    def test_a_molecular_reference_with_the_wrong_atom_count_is_refused(self):
        """E(N2) must be the energy of one molecule, or mu is off by a factor."""

        phases = dict(_TERNARY)
        phases["N2"] = {"composition": {"N": 4}, "energy_ev": 0.0}
        with self.assertRaises(SafetyError) as caught:
            map_report(phases, ["TiN"], ("N", "O"))
        self.assertIn("atoms", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
