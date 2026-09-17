from __future__ import annotations

import math
import unittest

from interfaceforge.errors import SafetyError
from interfaceforge.gas_reference import (
    EV_PER_KJ_PER_MOL,
    KB_EV_PER_K,
    PRESSURE_UNITS_IN_BAR,
    SPECIES,
    STANDARD_PRESSURE_BAR,
    THERMOCHEMICAL_SOURCE,
    enthalpy_above_0k_kj_per_mol,
    enthalpy_above_298_kj_per_mol,
    entropy_j_per_mol_k,
    gas_chemical_potential,
    gas_conditions,
    heat_capacity_j_per_mol_k,
    pressure_correction_ev,
    pressure_in_bar,
    reservoir_scheme,
    thermal_correction_ev,
)

#: Values NIST publishes in the table beside the Shomate coefficients, at
#: https://webbook.nist.gov (Chase 1998). Keyed (species, T) -> (Cp, S, H-H298).
#: These are the transcription check: a mistyped coefficient fails here rather
#: than quietly biasing every chemical potential downstream. The 300/400/500 K
#: rows sit in the first interval and 1000 K in the second, so both N2
#: intervals and both O2 intervals are exercised.
_NIST_TABLE = {
    ("N2", 300.0): (29.12, 191.8, 0.05),
    ("N2", 400.0): (29.25, 200.2, 2.97),
    ("N2", 500.0): (29.58, 206.7, 5.91),
    ("N2", 600.0): (30.10, 212.2, 8.89),
    ("N2", 1000.0): (32.69, 228.2, 21.46),
    ("O2", 300.0): (29.39, 205.3, 0.05),
    ("O2", 400.0): (30.10, 213.9, 3.03),
    ("O2", 500.0): (31.08, 220.7, 6.08),
    ("O2", 600.0): (32.11, 226.5, 9.24),
    ("O2", 1000.0): (34.86, 243.6, 22.70),
}

_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


class ShomateTest(unittest.TestCase):
    """The tabulated source data must come back out exactly as NIST prints it."""

    def test_every_published_value_is_reproduced(self):
        for (species, temperature), (cp, entropy, enthalpy) in _NIST_TABLE.items():
            with self.subTest(species=species, temperature=temperature):
                self.assertAlmostEqual(
                    heat_capacity_j_per_mol_k(species, temperature), cp, places=2
                )
                self.assertAlmostEqual(
                    entropy_j_per_mol_k(species, temperature), entropy, places=1
                )
                self.assertAlmostEqual(
                    enthalpy_above_298_kj_per_mol(species, temperature),
                    enthalpy, places=2,
                )

    def test_the_enthalpy_increment_vanishes_at_the_reference_temperature(self):
        """H(T) - H(298.15) is zero at 298.15 K by construction."""

        for species in ("N2", "O2"):
            self.assertAlmostEqual(
                enthalpy_above_298_kj_per_mol(species, 298.15), 0.0, places=3
            )

    def test_the_tabulated_standard_entropy_is_reproduced(self):
        """S(298.15, 1 bar) must match the CODATA value carried in the metadata."""

        for name, species in SPECIES.items():
            self.assertAlmostEqual(
                entropy_j_per_mol_k(name, 298.15),
                species.standard_entropy_j_per_mol_k,
                places=1,
            )

    def test_the_codata_0k_offset_is_reproduced_by_integrating_the_heat_capacity(self):
        """Cross-check the one number that is not itself a Shomate coefficient.

        H(298.15) - H(0) is taken from CODATA, so it deserves an independent
        check. Below 100 K a diatomic ideal gas has Cp = 7R/2 exactly (rotation
        fully active, vibration frozen -- and NIST tabulates 29.10 at 100 K,
        which is 7R/2 to four figures), so integrating that to 100 K and adding
        the Shomate increment from 100 K up must land on the CODATA value.
        """

        for name, species in SPECIES.items():
            with self.subTest(species=name):
                below_100k = 3.5 * _GAS_CONSTANT_J_PER_MOL_K * 100.0 / 1000.0
                above_100k = -enthalpy_above_298_kj_per_mol(name, 100.0)
                integrated = below_100k + above_100k
                # 0.02 kJ/mol covers the rounding in NIST's printed table plus
                # the rotational freeze-out below ~10 K that 7R/2 ignores.
                self.assertAlmostEqual(
                    integrated, species.h298_minus_h0_kj_per_mol, delta=0.02
                )

    def test_the_0k_referenced_enthalpy_adds_exactly_that_offset(self):
        for name, species in SPECIES.items():
            self.assertAlmostEqual(
                enthalpy_above_0k_kj_per_mol(name, 500.0)
                - enthalpy_above_298_kj_per_mol(name, 500.0),
                species.h298_minus_h0_kj_per_mol,
                places=12,
            )

    def test_a_temperature_outside_the_fitted_range_is_refused(self):
        """Extrapolating a Shomate fit returns a plausible number, so refuse instead."""

        with self.assertRaises(SafetyError) as caught:
            heat_capacity_j_per_mol_k("N2", 50.0)
        self.assertIn("outside the NIST Shomate range", str(caught.exception))
        with self.assertRaises(SafetyError):
            heat_capacity_j_per_mol_k("O2", 7000.0)

    def test_the_interval_boundaries_are_continuous(self):
        """Adjacent Shomate intervals must agree where they meet."""

        for name, species in SPECIES.items():
            for interval in species.intervals[:-1]:
                edge = interval.t_max
                with self.subTest(species=name, edge=edge):
                    # Approach from just inside each side of the boundary.
                    self.assertAlmostEqual(
                        entropy_j_per_mol_k(name, edge - 1.0e-6),
                        entropy_j_per_mol_k(name, edge + 1.0e-6),
                        places=2,
                    )

    def test_an_unknown_species_names_what_is_available(self):
        with self.assertRaises(SafetyError) as caught:
            heat_capacity_j_per_mol_k("CO2", 300.0)
        self.assertIn("no gas reference", str(caught.exception))


class PressureTest(unittest.TestCase):
    def test_no_pressure_correction_at_the_standard_state(self):
        """p = p0 is the definition of the standard state: the term is exactly zero."""

        for temperature in (100.0, 300.0, 1500.0):
            self.assertEqual(
                pressure_correction_ev(temperature, STANDARD_PRESSURE_BAR, "bar"), 0.0
            )

    def test_the_pressure_dependence_is_logarithmic(self):
        """Each decade of pressure costs the same kB*T*ln(10), per molecule."""

        temperature = 600.0
        decade = KB_EV_PER_K * temperature * math.log(10.0)
        previous = pressure_correction_ev(temperature, 1.0, "bar")
        for exponent in range(1, 8):
            current = pressure_correction_ev(temperature, 10.0 ** -exponent, "bar")
            self.assertAlmostEqual(previous - current, decade, places=12)
            previous = current

    def test_the_pressure_term_scales_linearly_with_temperature(self):
        a = pressure_correction_ev(300.0, 1.0e-10, "bar")
        b = pressure_correction_ev(600.0, 1.0e-10, "bar")
        self.assertAlmostEqual(b, 2.0 * a, places=12)

    def test_every_pressure_unit_converts_to_the_same_physical_pressure(self):
        """One atmosphere is one atmosphere however it is spelled."""

        one_atm_bar = 1.01325
        self.assertAlmostEqual(pressure_in_bar(1.0, "atm"), one_atm_bar, places=12)
        self.assertAlmostEqual(pressure_in_bar(101325.0, "Pa"), one_atm_bar, places=9)
        self.assertAlmostEqual(pressure_in_bar(101.325, "kPa"), one_atm_bar, places=9)
        self.assertAlmostEqual(pressure_in_bar(760.0, "torr"), one_atm_bar, places=9)
        self.assertAlmostEqual(pressure_in_bar(1013.25, "mbar"), one_atm_bar, places=9)
        self.assertAlmostEqual(pressure_in_bar(0.101325, "MPa"), one_atm_bar, places=9)

    def test_the_unit_reaches_the_chemical_potential(self):
        """A unit conversion that is only cosmetic would be worse than none."""

        in_atm = pressure_correction_ev(500.0, 1.0, "atm")
        in_bar = pressure_correction_ev(500.0, 1.01325, "bar")
        self.assertAlmostEqual(in_atm, in_bar, places=12)
        self.assertNotAlmostEqual(in_atm, pressure_correction_ev(500.0, 1.0, "bar"))

    def test_a_non_standard_reference_pressure_shifts_the_term(self):
        value = pressure_correction_ev(
            400.0, 1.0, "bar", standard_pressure_bar=1.01325
        )
        self.assertAlmostEqual(
            value, KB_EV_PER_K * 400.0 * math.log(1.0 / 1.01325), places=12
        )

    def test_nonpositive_pressure_is_refused(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(SafetyError) as caught:
                pressure_in_bar(bad, "bar")
            self.assertIn("must be positive", str(caught.exception))

    def test_nonpositive_temperature_is_refused(self):
        for bad in (0.0, -300.0):
            with self.assertRaises(SafetyError) as caught:
                pressure_correction_ev(bad, 1.0, "bar")
            self.assertIn("must be positive", str(caught.exception))

    def test_a_non_finite_input_is_refused(self):
        with self.assertRaises(SafetyError):
            pressure_in_bar(float("nan"), "bar")
        with self.assertRaises(SafetyError):
            pressure_correction_ev(float("inf"), 1.0, "bar")

    def test_an_unknown_unit_is_refused(self):
        with self.assertRaises(SafetyError) as caught:
            pressure_in_bar(1.0, "psi")
        self.assertIn("unknown pressure unit", str(caught.exception))
        self.assertIn("atm", str(caught.exception))


class FactorOfTwoTest(unittest.TestCase):
    """The molecule/atom conversion, pinned on its own.

    The 0 K hull measures dmu(N) from E(N2)/2, so every finite-T term must be
    halved on the way in. Getting this wrong by a factor of two would move the
    N-poor edge by an eV without any symptom other than a wrong answer.
    """

    def test_the_atomic_delta_is_exactly_half_the_molecular_one(self):
        for species in ("N2", "O2"):
            row = gas_chemical_potential(species, 700.0, 1.0e-8, unit="bar")
            self.assertEqual(row["atoms_per_molecule"], 2)
            self.assertAlmostEqual(
                row["dmu_atomic_ev"], row["dmu_molecular_ev"] / 2.0, places=15
            )
            self.assertAlmostEqual(
                row["pressure_correction_atomic_ev"],
                row["pressure_correction_ev"] / 2.0, places=15,
            )
            self.assertAlmostEqual(
                row["thermal_correction_atomic_ev"],
                row["thermal_correction_ev"] / 2.0, places=15,
            )

    def test_the_absolute_atomic_potential_is_half_the_molecular_one(self):
        row = gas_chemical_potential("O2", 600.0, 1.0e-12, unit="atm", energy_molecule_ev=-9.86)
        self.assertAlmostEqual(row["mu_atomic_ev"], row["mu_molecular_ev"] / 2.0, places=15)
        self.assertAlmostEqual(row["energy_atom_dft_ev"], -9.86 / 2.0, places=15)

    def test_the_dft_energy_cancels_out_of_the_delta(self):
        """dmu is measured from E(X2)/2, so adding E back in would double-count it.

        This is the bookkeeping trap in the whole layer: the absolute potential
        contains the DFT energy and the delta must not.
        """

        without = gas_chemical_potential("N2", 450.0, 0.2, unit="bar")
        for energy in (-16.0, -20.5, 0.0):
            with_energy = gas_chemical_potential(
                "N2", 450.0, 0.2, unit="bar", energy_molecule_ev=energy
            )
            self.assertAlmostEqual(
                with_energy["dmu_atomic_ev"], without["dmu_atomic_ev"], places=15
            )
            # and the absolute potential does move with it, by exactly E/2
            self.assertAlmostEqual(
                with_energy["mu_atomic_ev"] - energy / 2.0,
                without["dmu_atomic_ev"], places=12,
            )

    def test_the_thermal_correction_lowers_the_potential(self):
        """A gas gains entropy with temperature, so mu falls. Sign check."""

        previous = 0.0
        for temperature in (200.0, 400.0, 800.0, 1600.0):
            value = thermal_correction_ev("N2", temperature)
            self.assertLess(value, previous)
            previous = value


class ProvenanceTest(unittest.TestCase):
    def test_the_output_carries_everything_needed_to_reproduce_it(self):
        row = gas_chemical_potential(
            "O2", 600.0, 1.0e-15, unit="atm", energy_molecule_ev=-9.86
        )
        for key in (
            "temperature_k", "pressure_input", "pressure_unit", "pressure_bar",
            "standard_pressure_bar", "energy_molecule_dft_ev",
            "thermal_correction_ev", "pressure_correction_ev",
            "zero_point_energy_ev", "zero_point_energy_included",
            "empirical_correction_ev", "reference_scheme",
            "mu_molecular_ev", "mu_atomic_ev", "dmu_atomic_ev",
            "units", "thermochemical_source", "species_provenance", "equations",
            "solid_approximation",
        ):
            self.assertIn(key, row, f"{key} is missing from the output")
        self.assertIn("NIST-JANAF", row["thermochemical_source"]["shomate_reference"])
        self.assertIn("CODATA", row["thermochemical_source"]["h298_minus_h0_reference"])
        self.assertEqual(row["species_provenance"]["cas"], "7782-44-7")
        self.assertEqual(row["species_provenance"]["electronic_ground_state"], "X 3Sigma_g-")

    def test_the_payload_is_json_serialisable(self):
        import json

        payload = gas_conditions(600.0, {"N": 1.0, "O": 1.0e-15}, unit="atm")
        restored = json.loads(json.dumps(payload))
        self.assertEqual(
            restored["dmu_ev_per_atom"].keys(), payload["dmu_ev_per_atom"].keys()
        )

    def test_the_source_metadata_names_a_standard_state(self):
        self.assertEqual(
            THERMOCHEMICAL_SOURCE["standard_state_pressure_bar"], STANDARD_PRESSURE_BAR
        )
        self.assertEqual(STANDARD_PRESSURE_BAR, 1.0)

    def test_no_correction_is_applied_unless_one_is_named(self):
        """The default is the raw DFT reference, and the output says so."""

        row = gas_chemical_potential("O2", 300.0, 1.0, unit="bar")
        self.assertEqual(row["empirical_correction_ev"], 0.0)
        self.assertEqual(row["reference_scheme"], "raw-dft")
        self.assertFalse(row["zero_point_energy_included"])

    def test_an_unattributed_correction_is_refused(self):
        """A shift to a reference energy without a named scheme is unreproducible."""

        with self.assertRaises(SafetyError) as caught:
            gas_chemical_potential("O2", 300.0, 1.0, empirical_correction_ev=1.36)
        self.assertIn("reference_scheme", str(caught.exception))

    def test_a_named_correction_is_applied_and_recorded(self):
        plain = gas_chemical_potential("O2", 300.0, 1.0)
        shifted = gas_chemical_potential(
            "O2", 300.0, 1.0,
            empirical_correction_ev=1.36, reference_scheme="o2-overbinding-demo",
        )
        self.assertAlmostEqual(
            shifted["dmu_molecular_ev"] - plain["dmu_molecular_ev"], 1.36, places=12
        )
        # per atom it is half, like every other molecular term
        self.assertAlmostEqual(
            shifted["dmu_atomic_ev"] - plain["dmu_atomic_ev"], 0.68, places=12
        )
        self.assertEqual(shifted["empirical_correction_ev"], 1.36)
        self.assertEqual(shifted["reference_scheme"], "o2-overbinding-demo")

    def test_the_zero_point_energy_is_opt_in_but_always_reported(self):
        off = gas_chemical_potential("N2", 300.0, 1.0)
        on = gas_chemical_potential("N2", 300.0, 1.0, include_zero_point_energy=True)
        self.assertEqual(off["zero_point_energy_ev"], 0.0)
        self.assertGreater(off["zero_point_energy_available_ev"], 0.0)
        self.assertAlmostEqual(
            on["dmu_molecular_ev"] - off["dmu_molecular_ev"],
            off["zero_point_energy_available_ev"], places=12,
        )

    def test_the_zero_point_energies_match_the_published_constants(self):
        """ZPE = h*c*we_/2 with we_ from the NIST diatomic-constants tables."""

        self.assertAlmostEqual(SPECIES["N2"].vibrational_constant_cm1, 2358.57, places=2)
        self.assertAlmostEqual(SPECIES["O2"].vibrational_constant_cm1, 1580.193, places=3)
        self.assertAlmostEqual(SPECIES["N2"].zero_point_energy_ev, 0.14621, places=5)
        self.assertAlmostEqual(SPECIES["O2"].zero_point_energy_ev, 0.09796, places=5)


class ConditionsTest(unittest.TestCase):
    def test_a_condition_can_be_keyed_by_element_or_by_species(self):
        by_element = gas_conditions(500.0, {"N": 1.0, "O": 1.0e-9})
        by_species = gas_conditions(500.0, {"N2": 1.0, "O2": 1.0e-9})
        self.assertEqual(by_element["dmu_ev_per_atom"], by_species["dmu_ev_per_atom"])

    def test_the_reservoirs_are_independent(self):
        """Changing p_O2 must not move mu_N. They are separate reservoirs."""

        a = gas_conditions(600.0, {"N": 1.0, "O": 1.0e-5})
        b = gas_conditions(600.0, {"N": 1.0, "O": 1.0e-20})
        self.assertAlmostEqual(
            a["dmu_ev_per_atom"]["N"], b["dmu_ev_per_atom"]["N"], places=15
        )
        self.assertNotAlmostEqual(
            a["dmu_ev_per_atom"]["O"], b["dmu_ev_per_atom"]["O"]
        )

    def test_the_documented_magnitudes_hold(self):
        """The whole point of the layer: pressure moves dmu by ~eV, not ~meV.

        Pinned so that a units slip (kJ/mol left unconverted, or a factor of two)
        shows up as a wrong order of magnitude rather than a plausible number.
        """

        row = gas_conditions(600.0, {"N": 1.0, "O": 1.0e-20}, unit="atm")
        self.assertAlmostEqual(row["dmu_ev_per_atom"]["O"], -1.801, places=2)
        self.assertAlmostEqual(row["dmu_ev_per_atom"]["N"], -0.568, places=2)
        room = gas_conditions(300.0, {"O": 0.21}, unit="atm")
        self.assertAlmostEqual(room["dmu_ev_per_atom"]["O"], -0.294, places=2)


class ReservoirSchemeTest(unittest.TestCase):
    def test_the_direct_partial_pressure_scheme_is_implemented(self):
        self.assertTrue(reservoir_scheme("partial-pressure")["implemented"])

    def test_the_extension_points_refuse_with_the_equation_they_would_need(self):
        """A documented seam, not a silent fallback to the wrong physics."""

        with self.assertRaises(SafetyError) as caught:
            reservoir_scheme("h2-h2o")
        message = str(caught.exception)
        self.assertIn("extension point", message)
        self.assertIn("H2O", message)

    def test_an_unknown_scheme_is_refused(self):
        with self.assertRaises(SafetyError):
            reservoir_scheme("magic")


class ConstantsTest(unittest.TestCase):
    """The unit conversions are exact by SI definition; pin them as such."""

    def test_boltzmann_constant_in_ev_per_kelvin(self):
        self.assertAlmostEqual(KB_EV_PER_K, 8.617333262e-5, places=13)

    def test_kj_per_mol_to_ev(self):
        self.assertAlmostEqual(EV_PER_KJ_PER_MOL, 0.0103642697, places=10)

    def test_bar_is_the_reference_unit(self):
        self.assertEqual(PRESSURE_UNITS_IN_BAR["bar"], 1.0)
        self.assertEqual(PRESSURE_UNITS_IN_BAR["atm"], 1.01325)


if __name__ == "__main__":
    unittest.main()
