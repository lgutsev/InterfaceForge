# ruff: noqa: E501
"""Finite-temperature, finite-pressure chemical potentials for the N2 and O2 reservoirs.

The 0 K hull in :mod:`interfaceforge.phase_diagram` measures every chemical
potential from the elemental reference at 0 K: for nitrogen that is
``E(N2)/2``, the DFT total energy of one N2 molecule divided by its two atoms.
That reference is a *molecule at rest in a box*, which no experiment ever
provides. A real atmosphere sets the reservoir by temperature and partial
pressure, and this module is the bridge.

Everything here is about the **gas**. The solids keep their 0 K electronic
energies -- see ``SOLID_APPROXIMATION``. That is a deliberate, documented
approximation, not an oversight, and it is the reason nothing in this module is
called a Gibbs free energy of the interface.

The bookkeeping, per X2 molecule
--------------------------------

    mu_X2(T, p) = E_X2_dft [+ zpe] + dmu_thermal(T, p0) + dmu_pressure(T, p)

    dmu_thermal(T, p0) = [H(T) - H(0)] - T * S(T, p0)
    dmu_pressure(T, p) = kB * T * ln(p / p0)

and per **atom**, which is what the phase-diagram solver consumes:

    mu_X(T, p)  = mu_X2(T, p) / 2
    dmu_X(T, p) = mu_X(T, p) - E_X2_dft / 2
                = ([zpe] + dmu_thermal + dmu_pressure) / 2

Note what happens to ``E_X2_dft`` in the last line: it **cancels exactly**.
The 0 K scale is already measured from ``E_X2_dft/2``, so adding the DFT energy
back into a *delta* would count it twice. ``mu_molecular_ev`` and
``mu_atomic_ev`` in the output are the absolute potentials and do contain it;
``dmu_atomic_ev`` is the one the hull consumes and does not. The factor of two
sits in exactly one place per quantity, and :mod:`tests.test_gas_reference`
pins it.

Where the numbers come from
---------------------------

Heat capacity, entropy and the enthalpy increment above 298.15 K come from the
Shomate parameterisation published in the NIST Chemistry WebBook, which
reproduces Chase, M. W., Jr., *NIST-JANAF Thermochemical Tables, Fourth
Edition*, J. Phys. Chem. Ref. Data, Monograph 9 (1998). Coefficients were
transcribed from the WebBook species pages on 2026-09-16 and every interval is
regression-tested against the tabulated Cp, S and H values NIST publishes
alongside them (see ``tests/test_gas_reference.py``), so a transcription slip
fails the suite rather than quietly biasing a chemical potential.

The Shomate enthalpy is referenced to **298.15 K**, while a DFT total energy is
a 0 K quantity. The conversion is ``H(298.15) - H(0)``, taken from the CODATA
Key Values (Cox, Wagman and Medvedev, 1989) and cross-checked here by
integrating the NIST heat capacity down to 0 K. Skipping it would shift
dmu_N by 0.045 eV/atom and dmu_O by the same, which is not a rounding error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import SafetyError

# --------------------------------------------------------------- constants

#: Boltzmann constant in eV/K. Exact: the SI defines kB = 1.380649e-23 J/K and
#: the elementary charge e = 1.602176634e-19 C, so this ratio carries no
#: experimental uncertainty at all.
KB_EV_PER_K = 1.380649e-23 / 1.602176634e-19

#: Faraday constant in J/(mol*V) = N_A * e, also exact by SI definition. One
#: kJ/mol is 1000 / F electronvolts per particle.
FARADAY_J_PER_MOL_V = 6.02214076e23 * 1.602176634e-19

#: eV per kJ/mol.
EV_PER_KJ_PER_MOL = 1000.0 / FARADAY_J_PER_MOL_V

#: cm^-1 to eV, for the zero-point energies derived from vibrational constants.
#: h*c/e with h = 6.62607015e-34 J s (exact) and c = 299792458 m/s (exact).
EV_PER_WAVENUMBER = 6.62607015e-34 * 299792458.0 * 100.0 / 1.602176634e-19

#: Standard-state pressure of the tabulated entropies, in bar. NIST-JANAF
#: tabulates S at 1 bar (the WebBook column is labelled "S(gas, 1 bar)"), which
#: is the post-1982 IUPAC standard state. Using 1 atm here instead would bias
#: every entropy term by kB*T*ln(1.01325) -- small, but wrong, and silent.
STANDARD_PRESSURE_BAR = 1.0

#: Accepted pressure units and their value in bar. 1 atm = 101325 Pa exactly
#: (the definition), 1 torr = 1 atm / 760 exactly.
PRESSURE_UNITS_IN_BAR = {
    "bar": 1.0,
    "mbar": 1.0e-3,
    "Pa": 1.0e-5,
    "kPa": 1.0e-2,
    "MPa": 10.0,
    "atm": 1.01325,
    "torr": 1.01325 / 760.0,
}

SOLID_APPROXIMATION = (
    "Solid phases keep their 0 K DFT electronic energies: no phonon, thermal "
    "expansion or configurational term is applied to them. Only the gas "
    "reservoirs carry temperature and pressure. This is the standard ab-initio "
    "thermodynamics approximation and it is good precisely because the gas "
    "entropy term dominates -- roughly 0.6 eV per O2 at 600 K against a few "
    "tens of meV for a solid -- but it means these are NOT full finite-"
    "temperature Gibbs free energies, and a boundary between two solids moves "
    "less than this treatment can resolve."
)


# ------------------------------------------------------------ source data

@dataclass(frozen=True)
class ShomateInterval:
    """One temperature interval of a NIST Shomate parameterisation.

    ``t_min``/``t_max`` are in K; the coefficients are NIST's, for ``t = T/1000``
    with Cp in J/(mol K), S in J/(mol K) and H in kJ/mol.
    """

    t_min: float
    t_max: float
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    g: float
    h: float


@dataclass(frozen=True)
class GasSpecies:
    """A diatomic gas reservoir and everything needed to place it on the 0 K scale."""

    name: str
    element: str
    atoms_per_molecule: int
    intervals: tuple[ShomateInterval, ...]
    #: H(298.15) - H(0) in kJ/mol, with its stated uncertainty. CODATA Key Values.
    h298_minus_h0_kj_per_mol: float
    h298_minus_h0_uncertainty_kj_per_mol: float
    #: Harmonic zero-point energy in eV, from the NIST vibrational constant we_.
    zero_point_energy_ev: float
    #: we_ in cm^-1 and the electronic ground state it belongs to, for provenance.
    vibrational_constant_cm1: float
    electronic_ground_state: str
    #: S(298.15 K, 1 bar) in J/(mol K), CODATA, used only as a self-check.
    standard_entropy_j_per_mol_k: float
    cas: str
    source_url: str


def _zpe_ev(wavenumber_cm1: float) -> float:
    """Harmonic zero-point energy of a diatomic, ``hc*we_/2``.

    Harmonic on purpose: the anharmonic G(0) = we_/2 - wexe_/4 + ... lowers N2
    by 0.4 meV and O2 by 0.4 meV, far below the DFT error on the same molecule,
    and the harmonic form is the one a frequency calculation on the solid would
    also use, so the two stay comparable.
    """

    return 0.5 * wavenumber_cm1 * EV_PER_WAVENUMBER


#: Transcribed from the NIST Chemistry WebBook on 2026-09-16. The Shomate
#: coefficients, the vibrational constants and the CODATA entropies each carry
#: their own page; ``source_url`` points at the species record.
SPECIES: dict[str, GasSpecies] = {
    "N2": GasSpecies(
        name="N2",
        element="N",
        atoms_per_molecule=2,
        intervals=(
            ShomateInterval(100.0, 500.0, 28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914, 226.4168, 0.0),
            ShomateInterval(500.0, 2000.0, 19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 212.3900, 0.0),
            ShomateInterval(2000.0, 6000.0, 35.51872, 1.128728, -0.196103, 0.014662, -4.553760, -18.97091, 224.9810, 0.0),
        ),
        h298_minus_h0_kj_per_mol=8.670,
        h298_minus_h0_uncertainty_kj_per_mol=0.001,
        zero_point_energy_ev=_zpe_ev(2358.57),
        vibrational_constant_cm1=2358.57,
        electronic_ground_state="X 1Sigma_g+",
        standard_entropy_j_per_mol_k=191.609,
        cas="7727-37-9",
        source_url="https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Units=SI&Mask=1",
    ),
    "O2": GasSpecies(
        name="O2",
        element="O",
        atoms_per_molecule=2,
        intervals=(
            ShomateInterval(100.0, 700.0, 31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945, 0.0),
            ShomateInterval(700.0, 2000.0, 30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663, 0.0),
            ShomateInterval(2000.0, 6000.0, 20.91111, 10.72071, -2.020498, 0.146449, 9.245722, 5.337651, 237.6185, 0.0),
        ),
        h298_minus_h0_kj_per_mol=8.680,
        h298_minus_h0_uncertainty_kj_per_mol=0.002,
        zero_point_energy_ev=_zpe_ev(1580.193),
        vibrational_constant_cm1=1580.193,
        electronic_ground_state="X 3Sigma_g-",
        standard_entropy_j_per_mol_k=205.152,
        cas="7782-44-7",
        source_url="https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Units=SI&Mask=1",
    ),
}

#: Which gas sets the potential of which element. Keyed explicitly rather than
#: sliced off the formula, so asking for Na does not pull in N2.
ELEMENT_TO_SPECIES = {species.element: name for name, species in SPECIES.items()}

THERMOCHEMICAL_SOURCE = {
    "dataset": "nist-janaf-shomate",
    "shomate_reference": (
        "Chase, M. W., Jr., NIST-JANAF Thermochemical Tables, Fourth Edition, "
        "J. Phys. Chem. Ref. Data, Monograph 9 (1998), 1-1951, as published in "
        "the NIST Chemistry WebBook (SRD 69)."
    ),
    "h298_minus_h0_reference": (
        "Cox, J. D.; Wagman, D. D.; Medvedev, V. A., CODATA Key Values for "
        "Thermodynamics, Hemisphere Publishing Corp., New York (1989)."
    ),
    "vibrational_constant_reference": (
        "Huber, K. P.; Herzberg, G., Molecular Spectra and Molecular Structure "
        "IV: Constants of Diatomic Molecules, as published in the NIST "
        "Chemistry WebBook (SRD 69)."
    ),
    "standard_state_pressure_bar": STANDARD_PRESSURE_BAR,
    "transcribed": "2026-09-16",
    "note": (
        "Shomate enthalpies are referenced to 298.15 K; the CODATA "
        "H(298.15)-H(0) increment moves them onto the 0 K scale of a DFT total "
        "energy. Every interval is regression-tested against the Cp, S and H "
        "values NIST tabulates on the same page."
    ),
}


# ----------------------------------------------------------- the equations

def _species(name_or_element: str) -> GasSpecies:
    if name_or_element in SPECIES:
        return SPECIES[name_or_element]
    if name_or_element in ELEMENT_TO_SPECIES:
        return SPECIES[ELEMENT_TO_SPECIES[name_or_element]]
    raise SafetyError(
        f"no gas reference for {name_or_element!r}; this module carries "
        f"{sorted(SPECIES)} (elements {sorted(ELEMENT_TO_SPECIES)}). Adding one "
        "means transcribing its Shomate intervals and its CODATA "
        "H(298.15)-H(0), and pinning both against the published table."
    )


def _interval(species: GasSpecies, temperature_k: float) -> ShomateInterval:
    """The Shomate interval covering this temperature, or a refusal.

    Extrapolating a Shomate fit outside its range is exactly the kind of quiet
    wrongness the house rules forbid: the polynomial stays finite and plausible
    while the physics it was fitted to has ended.
    """

    for interval in species.intervals:
        if interval.t_min <= temperature_k <= interval.t_max:
            return interval
    low = species.intervals[0].t_min
    high = species.intervals[-1].t_max
    raise SafetyError(
        f"T = {temperature_k} K is outside the NIST Shomate range for "
        f"{species.name} ({low:g}-{high:g} K). The polynomial would still return "
        "a number, which is why this refuses instead: outside the fitted range "
        "it is not the JANAF data any more."
    )


def check_temperature(temperature_k: float) -> float:
    """Reject a non-physical temperature before it reaches a logarithm."""

    value = float(temperature_k)
    if not math.isfinite(value):
        raise SafetyError(f"temperature must be finite; got {temperature_k!r}")
    if value <= 0.0:
        raise SafetyError(
            f"temperature must be positive; got {value} K. The gas entropy term "
            "is T*S and the pressure term is kB*T*ln(p/p0); neither is defined "
            "at or below 0 K."
        )
    return value


def pressure_in_bar(pressure: float, unit: str = "bar") -> float:
    """Convert a partial pressure to bar, rejecting non-physical values."""

    if unit not in PRESSURE_UNITS_IN_BAR:
        raise SafetyError(
            f"unknown pressure unit {unit!r}; known units are "
            f"{sorted(PRESSURE_UNITS_IN_BAR)}"
        )
    value = float(pressure)
    if not math.isfinite(value):
        raise SafetyError(f"pressure must be finite; got {pressure!r}")
    if value <= 0.0:
        raise SafetyError(
            f"pressure must be positive; got {value} {unit}. kB*T*ln(p/p0) "
            "diverges as p -> 0: a perfectly O-free reservoir is dmu_O -> -inf, "
            "not p_O2 = 0. Use a small positive pressure, or drop the element "
            "from the reservoir entirely."
        )
    return value * PRESSURE_UNITS_IN_BAR[unit]


def heat_capacity_j_per_mol_k(species: str, temperature_k: float) -> float:
    """Shomate Cp(T) in J/(mol K). Pins directly against the NIST table."""

    gas = _species(species)
    temperature = check_temperature(temperature_k)
    i = _interval(gas, temperature)
    t = temperature / 1000.0
    return i.a + i.b * t + i.c * t * t + i.d * t * t * t + i.e / (t * t)


def entropy_j_per_mol_k(species: str, temperature_k: float) -> float:
    """Shomate S(T) at the standard-state pressure, in J/(mol K)."""

    gas = _species(species)
    temperature = check_temperature(temperature_k)
    i = _interval(gas, temperature)
    t = temperature / 1000.0
    return (
        i.a * math.log(t)
        + i.b * t
        + i.c * t * t / 2.0
        + i.d * t * t * t / 3.0
        - i.e / (2.0 * t * t)
        + i.g
    )


def enthalpy_above_298_kj_per_mol(species: str, temperature_k: float) -> float:
    """Shomate H(T) - H(298.15 K) in kJ/mol, exactly as NIST defines it."""

    gas = _species(species)
    temperature = check_temperature(temperature_k)
    i = _interval(gas, temperature)
    t = temperature / 1000.0
    return (
        i.a * t
        + i.b * t * t / 2.0
        + i.c * t * t * t / 3.0
        + i.d * t * t * t * t / 4.0
        - i.e / t
        + i.f
        - i.h
    )


def enthalpy_above_0k_kj_per_mol(species: str, temperature_k: float) -> float:
    """H(T) - H(0) in kJ/mol: the Shomate increment put on the 0 K scale.

    This is the conversion that makes the tabulated data commensurate with a
    DFT total energy. It excludes the zero-point energy by construction, since
    the ZPE sits inside H(0) and cancels in the difference -- which is why the
    ZPE is a separate, explicit term here rather than folded in silently.
    """

    gas = _species(species)
    return enthalpy_above_298_kj_per_mol(species, temperature_k) + gas.h298_minus_h0_kj_per_mol


def thermal_correction_ev(species: str, temperature_k: float) -> float:
    """dmu_thermal(T, p0) = [H(T) - H(0)] - T*S(T), per molecule, in eV.

    Negative and growing in magnitude with T: the gas gains entropy, so its
    chemical potential falls below the 0 K molecule.
    """

    temperature = check_temperature(temperature_k)
    enthalpy_ev = enthalpy_above_0k_kj_per_mol(species, temperature) * EV_PER_KJ_PER_MOL
    entropy_ev_per_k = entropy_j_per_mol_k(species, temperature) * EV_PER_KJ_PER_MOL / 1000.0
    return enthalpy_ev - temperature * entropy_ev_per_k


def pressure_correction_ev(
    temperature_k: float,
    pressure: float,
    unit: str = "bar",
    *,
    standard_pressure_bar: float = STANDARD_PRESSURE_BAR,
) -> float:
    """kB*T*ln(p/p0) per molecule, in eV. Ideal gas, which at these pressures it is."""

    temperature = check_temperature(temperature_k)
    p_bar = pressure_in_bar(pressure, unit)
    p0 = float(standard_pressure_bar)
    if not math.isfinite(p0) or p0 <= 0.0:
        raise SafetyError(f"standard-state pressure must be positive; got {p0!r} bar")
    return KB_EV_PER_K * temperature * math.log(p_bar / p0)


def gas_chemical_potential(
    species: str,
    temperature_k: float,
    pressure: float,
    *,
    unit: str = "bar",
    energy_molecule_ev: float | None = None,
    standard_pressure_bar: float = STANDARD_PRESSURE_BAR,
    include_zero_point_energy: bool = False,
    reference_scheme: str = "raw-dft",
    empirical_correction_ev: float = 0.0,
) -> dict[str, Any]:
    """Everything about one gas reservoir at (T, p), with its provenance attached.

    ``energy_molecule_ev`` is the DFT total energy of the X2 molecule, the same
    number handed to the hull as that element's reference. Supply it and the
    absolute potentials are filled in; omit it and only the deltas are, which
    are the part the phase-diagram solver actually consumes.

    ``empirical_correction_ev`` is applied **per molecule** and is zero unless
    a caller names it. It exists because an O2 reference computed in GGA is
    known to be overbound, and some workflows correct for that -- but a fitted
    shift is a choice about the data, not a property of it, so it is opt-in,
    it is echoed in the output, and ``reference_scheme`` records which scheme
    the number came from. Nothing in this module invents one.
    """

    gas = _species(species)
    temperature = check_temperature(temperature_k)
    p_bar = pressure_in_bar(pressure, unit)
    correction = float(empirical_correction_ev)
    if not math.isfinite(correction):
        raise SafetyError("empirical correction must be finite")
    if correction != 0.0 and reference_scheme == "raw-dft":
        raise SafetyError(
            f"a non-zero empirical correction ({correction} eV) was supplied while "
            "reference_scheme is still 'raw-dft'. Name the scheme the number came "
            "from so the output records it -- an unattributed shift to a reference "
            "energy is the one thing that makes a chemical potential unreproducible."
        )

    thermal = thermal_correction_ev(gas.name, temperature)
    pressure_term = pressure_correction_ev(
        temperature, pressure, unit, standard_pressure_bar=standard_pressure_bar
    )
    zpe = gas.zero_point_energy_ev if include_zero_point_energy else 0.0
    n = gas.atoms_per_molecule

    # The delta the hull consumes. E_dft cancels out of it by construction --
    # the 0 K scale is already measured from E_dft/n.
    delta_molecular = zpe + thermal + pressure_term + correction
    delta_atomic = delta_molecular / n

    payload: dict[str, Any] = {
        "species": gas.name,
        "element": gas.element,
        "atoms_per_molecule": n,
        "temperature_k": temperature,
        "pressure_input": float(pressure),
        "pressure_unit": unit,
        "pressure_bar": p_bar,
        "standard_pressure_bar": float(standard_pressure_bar),
        "energy_molecule_dft_ev": None if energy_molecule_ev is None else float(energy_molecule_ev),
        "thermal_correction_ev": thermal,
        "pressure_correction_ev": pressure_term,
        "zero_point_energy_ev": zpe,
        "zero_point_energy_included": bool(include_zero_point_energy),
        "zero_point_energy_available_ev": gas.zero_point_energy_ev,
        "empirical_correction_ev": correction,
        "reference_scheme": reference_scheme,
        "dmu_molecular_ev": delta_molecular,
        "dmu_atomic_ev": delta_atomic,
        "pressure_correction_atomic_ev": pressure_term / n,
        "thermal_correction_atomic_ev": thermal / n,
        "mu_molecular_ev": None,
        "mu_atomic_ev": None,
        "units": {
            "temperature_k": "K",
            "pressure_bar": "bar",
            "energy_molecule_dft_ev": "eV per molecule",
            "thermal_correction_ev": "eV per molecule",
            "pressure_correction_ev": "eV per molecule",
            "dmu_molecular_ev": "eV per molecule",
            "dmu_atomic_ev": "eV per atom, relative to E_dft(X2)/2",
            "mu_molecular_ev": "eV per molecule, absolute",
            "mu_atomic_ev": "eV per atom, absolute",
        },
        "thermochemical_source": dict(THERMOCHEMICAL_SOURCE),
        "species_provenance": {
            "cas": gas.cas,
            "source_url": gas.source_url,
            "electronic_ground_state": gas.electronic_ground_state,
            "vibrational_constant_cm1": gas.vibrational_constant_cm1,
            "h298_minus_h0_kj_per_mol": gas.h298_minus_h0_kj_per_mol,
            "h298_minus_h0_uncertainty_kj_per_mol": gas.h298_minus_h0_uncertainty_kj_per_mol,
        },
        "equations": {
            "mu_molecular": "mu_X2(T,p) = E_X2_dft [+ zpe] + dmu_thermal(T,p0) + kB*T*ln(p/p0) [+ empirical]",
            "dmu_thermal": "dmu_thermal(T,p0) = [H(T)-H(0)] - T*S(T,p0), Shomate + CODATA H(298.15)-H(0)",
            "dmu_pressure": "dmu_pressure(T,p) = kB*T*ln(p/p0)",
            "atomic": "mu_X = mu_X2 / 2; dmu_X = mu_X - E_X2_dft/2 = dmu_molecular / 2",
        },
        "solid_approximation": SOLID_APPROXIMATION,
    }
    if energy_molecule_ev is not None:
        energy = float(energy_molecule_ev)
        if not math.isfinite(energy):
            raise SafetyError("the molecular DFT energy must be finite")
        payload["mu_molecular_ev"] = energy + delta_molecular
        payload["mu_atomic_ev"] = (energy + delta_molecular) / n
        payload["energy_atom_dft_ev"] = energy / n
    return payload


def gas_conditions(
    temperature_k: float,
    pressures: dict[str, float],
    *,
    unit: str = "bar",
    energies_ev: dict[str, float] | None = None,
    standard_pressure_bar: float = STANDARD_PRESSURE_BAR,
    include_zero_point_energy: bool = False,
    reference_scheme: str = "raw-dft",
    empirical_corrections_ev: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One (T, p_N2, p_O2, ...) condition, keyed by **element**.

    ``pressures`` is keyed by element ("N", "O") or by species ("N2", "O2");
    either spelling resolves to the same reservoir. ``energies_ev`` is keyed the
    same way and holds the DFT energy of the molecule, not of the atom.
    """

    energies = dict(energies_ev or {})
    corrections = dict(empirical_corrections_ev or {})
    per_element: dict[str, Any] = {}
    for key, pressure in pressures.items():
        gas = _species(key)
        energy = energies.get(gas.element, energies.get(gas.name))
        correction = corrections.get(gas.element, corrections.get(gas.name, 0.0))
        per_element[gas.element] = gas_chemical_potential(
            gas.name, temperature_k, pressure,
            unit=unit,
            energy_molecule_ev=energy,
            standard_pressure_bar=standard_pressure_bar,
            include_zero_point_energy=include_zero_point_energy,
            reference_scheme=reference_scheme,
            empirical_correction_ev=correction,
        )
    return {
        "schema_version": 1,
        "temperature_k": check_temperature(temperature_k),
        "pressure_unit": unit,
        "standard_pressure_bar": float(standard_pressure_bar),
        "reference_scheme": reference_scheme,
        "zero_point_energy_included": bool(include_zero_point_energy),
        "reservoirs": per_element,
        "dmu_ev_per_atom": {
            element: row["dmu_atomic_ev"] for element, row in per_element.items()
        },
        "thermochemical_source": dict(THERMOCHEMICAL_SOURCE),
        "solid_approximation": SOLID_APPROXIMATION,
        "atmosphere_note": ATMOSPHERE_NOTE,
    }


# ------------------------------------------------- what "anoxic" actually means

ATMOSPHERE_NOTE = (
    "A partial pressure is an *equilibrated O2 reservoir*, which is a stronger "
    "statement than the name of a gas bottle. Nitrogen, argon and vacuum do not "
    "come with a p_O2: they come with a residual oxygen content that is set by "
    "leak rate, outgassing and getter chemistry, and that is a kinetic quantity, "
    "not a thermodynamic one. Quoting p_O2 = 1e-20 atm for 'an argon glovebox' "
    "asserts an equilibrium nobody measured. Use the value you can defend -- an "
    "oxygen analyser reading, or a bracketing range -- and treat the rest as a "
    "sensitivity scan."
)

#: Named reservoir schemes, and what each one would need. Only the direct
#: partial pressure is implemented; the rest are the seam, documented so the
#: next person implements the right equation rather than guessing one.
RESERVOIR_SCHEMES = {
    "partial-pressure": {
        "implemented": True,
        "description": (
            "mu_O set directly by an equilibrated O2 partial pressure. Use when "
            "p_O2 is measured, or as an explicit sensitivity parameter."
        ),
    },
    "h2-h2o": {
        "implemented": False,
        "description": (
            "mu_O from the H2 + 1/2 O2 <-> H2O equilibrium, which is how a "
            "reducing forming-gas atmosphere actually buffers oxygen: "
            "mu_O(T) = mu_H2O(T,p_H2O) - mu_H2(T,p_H2). In the notation here "
            "that is dmu_O = dG_f(H2O,T) + kB*T*ln(p_H2O/p_H2), with dG_f from "
            "the same JANAF source. Needs the H2 and H2O Shomate intervals plus "
            "the formation enthalpy of H2O, transcribed and pinned the same way."
        ),
    },
    "co-co2": {
        "implemented": False,
        "description": (
            "mu_O from the CO + 1/2 O2 <-> CO2 buffer, the usual choice above "
            "about 1000 K where H2/H2O is less convenient. Same shape, needs "
            "the CO and CO2 data."
        ),
    },
}


def reservoir_scheme(name: str) -> dict[str, Any]:
    """Look up a named reservoir scheme, refusing the ones not yet implemented."""

    if name not in RESERVOIR_SCHEMES:
        raise SafetyError(
            f"unknown reservoir scheme {name!r}; known schemes are "
            f"{sorted(RESERVOIR_SCHEMES)}"
        )
    scheme = RESERVOIR_SCHEMES[name]
    if not scheme["implemented"]:
        raise SafetyError(
            f"reservoir scheme {name!r} is a documented extension point, not an "
            f"implementation. {scheme['description']} Until then, supply the "
            "equivalent p_O2 directly with --p-o2 and say in the caption that it "
            "is an effective value."
        )
    return dict(scheme)
