#!/usr/bin/env python
"""Plot the planar-averaged LOCPOT potential and estimate the work function.

Standalone, dependency-light (ase + numpy + matplotlib only) script meant to
run unattended at the end of a surface-optimization VASP job, e.g. from
``/home/$USER/bin`` on LONI, invoked directly by ``runvasp.sh``. It does not
depend on InterfaceForge being installed in the job's environment.

Requires ``LVHAR = .TRUE.`` in the INCAR that produced LOCPOT: VASP only
writes the electrostatic potential (rather than just the charge density)
into LOCPOT when that tag is set. Without it, this script produces numbers
that look plausible but are not the work function.

When InterfaceForge itself is available (e.g. from a controller or analysis
environment), prefer ``iface vasp workfunction LOCPOT OUTCAR``
(``interfaceforge.workfunction.analyze_workfunction``) instead: it is unit
tested and additionally writes a JSON summary. This script exists for the
case where that environment is not present on the compute node.
"""
import logging
import os
import re
import sys
from argparse import ArgumentParser

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.calculators.vasp import VaspChargeDensity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VASP_PROTECTED_NAMES = {
    "CHG", "CHGCAR", "CONTCAR", "DOSCAR", "EIGENVAL", "IBZKPT", "INCAR",
    "KPOINTS", "LOCPOT", "OSZICAR", "OUTCAR", "PCDAT", "POSCAR", "POTCAR",
    "PROCAR", "VASPRUN.XML", "WAVECAR", "WAVEDER", "XDATCAR",
}


def safe_output(path):
    """Reject output names that could damage a VASP calculation or restart."""

    if os.path.islink(path):
        raise ValueError(f"Refusing to overwrite symlinked output: {path}")
    if os.path.basename(path).upper() in VASP_PROTECTED_NAMES:
        raise ValueError(f"Refusing to overwrite protected VASP file: {path}")
    return os.path.realpath(path)


def check_lvhar(incar="INCAR"):
    """Return True if INCAR declares LVHAR = True/.TRUE./T.

    Returns None (rather than False) when INCAR cannot be read, since the
    absence of an INCAR alongside LOCPOT is a different, ambiguous, problem
    that the caller should decide how to handle.
    """
    if not os.path.isfile(incar):
        logger.warning("%s not found; cannot confirm LVHAR was set", incar)
        return None
    text = open(incar).read()
    match = re.search(r"(?im)^\s*LVHAR\s*=\s*\.?(TRUE|T)\.?", text)
    return bool(match)


def locpot_mean(fname="LOCPOT", axis="z", savefile="locpot.dat", outcar="OUTCAR"):
    """Read LOCPOT and calculate the planar-averaged potential along `axis`.

    @out:
        - xvals: grid coordinate along the selected axis (Angstrom);
        - mean: planar-averaged potential (eV) corresponding to `xvals`,
          shifted so that 0 eV is the Fermi level when OUTCAR is available.
    """

    def get_efermi(outcar="OUTCAR"):
        if not os.path.isfile(outcar):
            logger.warning("OUTCAR file not found. E-fermi set to 0.0eV")
            return None
        txt = open(outcar).read()
        matches = re.findall(
            r"E-fermi\s*:\s*([-+]?[0-9]+[.]?[0-9]*(?:[eE][-+]?[0-9]+)?)", txt
        )
        if not matches:
            logger.warning("No E-fermi line found in OUTCAR. E-fermi set to 0.0eV")
            return None
        efermi = matches[-1]
        logger.info(f"Found E-fermi = {efermi}")
        return float(efermi)

    logger.info(f"Loading LOCPOT file {fname}")
    locd = VaspChargeDensity(fname)
    cell = locd.atoms[0].cell
    latlens = np.linalg.norm(cell, axis=1)
    # ASE's VaspChargeDensity divides every raw LOCPOT grid value by
    # atoms.get_volume() == abs(det(cell)). Multiply back by the same
    # unsigned volume; using the signed determinant would flip the sign of
    # the recovered potential for a left-handed cell.
    vol = float(abs(np.linalg.det(cell)))

    iaxis = ["x", "y", "z"].index(axis.lower())
    axes = tuple(index for index in (0, 1, 2) if index != iaxis)

    locpot = locd.chg[0]
    logger.info(f"Calculating workfunction along {axis} axis")
    mean = np.mean(locpot, axis=axes) * vol

    xvals = np.linspace(0, latlens[iaxis], locpot.shape[iaxis], endpoint=False)

    efermi = get_efermi(outcar)
    logger.info(f"Saving raw data to {savefile}")
    if efermi is None:
        np.savetxt(
            savefile,
            np.c_[xvals, mean],
            fmt="%13.5f",
            header="Distance(A) Potential(eV) # E-fermi not corrected",
        )
    else:
        mean = mean - efermi
        np.savetxt(
            savefile,
            np.c_[xvals, mean],
            fmt="%13.5f",
            header="Distance(A) Potential(eV) # E-fermi shifted to 0.0eV",
        )
    return (xvals, mean)


def parse_cml_arguments(argv=None):
    parser = ArgumentParser(
        description="A tool to plot work function according to LOCPOT", add_help=True
    )
    parser.add_argument(
        "-a", "--axis", type=str, action="store",
        help="Which axis to be calculated: x, y or z. Default by z",
        default="z", choices=["x", "y", "z"],
    )
    parser.add_argument(
        "input", nargs="?", type=str,
        help="The input file name, default by LOCPOT", default="LOCPOT",
    )
    parser.add_argument(
        "--incar", type=str, action="store",
        help="INCAR used to confirm LVHAR = True, default by INCAR", default="INCAR",
    )
    parser.add_argument(
        "-w", "--write", type=str, action="store",
        help="Save raw work function data to file, default by locpot.dat",
        default="locpot.dat",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store",
        help="Output image file name, default by Workfunction.png",
        default="Workfunction.png",
    )
    parser.add_argument(
        "--dpi", type=int, action="store",
        help="DPI of output image, default by 400", default=400,
    )
    parser.add_argument(
        "--title", type=str, action="store",
        help="Title in output image. If none, no title is added, default is None",
        default=None,
    )
    parser.add_argument(
        "--allow-missing-lvhar", action="store_true",
        help="Plot even if LVHAR = True could not be confirmed in --incar "
        "(default: refuse, since the LOCPOT contents would not be the "
        "electrostatic potential)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_cml_arguments(argv)

    try:
        args.write = safe_output(args.write)
        args.output = safe_output(args.output)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    lvhar = check_lvhar(args.incar)
    if lvhar is False and not args.allow_missing_lvhar:
        logger.error(
            "LVHAR = True was not found in %s. Without it, LOCPOT holds the "
            "charge density, not the electrostatic potential, and the plotted "
            "work function would be meaningless. Re-run with LVHAR = .TRUE. "
            "in INCAR, or pass --allow-missing-lvhar to override.",
            args.incar,
        )
        return 1

    x, y = locpot_mean(args.input, args.axis, args.write, outcar="OUTCAR")

    logger.info("Plotting to image")
    plt.plot(x, y, color="k")
    plt.xlabel("Distance(A)")
    plt.ylabel("Potential(eV)")
    plt.grid(color="gray", ls="-.")
    plt.xlim(0, np.max(x))
    plt.ylim(np.max(y) - 2, np.max(y) + 0.5)
    plt.minorticks_on()

    if args.title:
        plt.title(args.title)

    logger.info(f"Saving to {args.output}")
    plt.savefig(args.output, dpi=args.dpi)
    return 0


if "__main__" == __name__:
    sys.exit(main())
