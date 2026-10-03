"""Launch-time integration: run the initializer inside the VASP job, just before VASP.

This is the HPC-safe default.  Preparation nodes often lack the GPU, CUDA
toolkit, network (for first-use weight download) or Python environment the
initializer needs, so ``step1-prepare --density-init`` and the benchmark only
*record* the request and wrap the launcher; inference happens on the compute
node immediately before VASP, on the exact inputs VASP will read.

Failure policy: ``standard`` (default) lets VASP continue with its normal
atomic-density start when initialization fails -- the initializer guarantees
the inputs are untouched in that case -- so a broken ML stack never wastes an
allocation.  ``abort`` stops the job instead (used by the benchmark so that an
initializer failure is counted, never hidden).
"""

from __future__ import annotations

import shlex
import sys

from ..errors import SafetyError
from ..vasp import _PRECONDITION_MARKER, _VASP_INVOCATION

DENSITY_INIT_MARKER = "InterfaceForge --density-init"
ON_FAILURE = ("standard", "abort")
_PRECONDITION_STEP = "mv -f INCAR.precondition INCAR && "


def default_interfaceforge_command() -> str:
    return f"{shlex.quote(sys.executable)} -m interfaceforge"


def vasp_command_from_launcher(text: str) -> str | None:
    hits = [line.strip() for line in text.splitlines() if _VASP_INVOCATION.match(line)]
    return hits[-1] if hits else None


def wrap_launcher_with_density_init(
    launcher_text: str, *, launcher_name: str, hook: str, on_failure: str = "standard"
) -> str:
    """Insert ``hook`` so it runs right before the fresh-start VASP call."""

    if on_failure not in ON_FAILURE:
        raise SafetyError(f"--density-init-on-failure must be one of {', '.join(ON_FAILURE)}")
    if DENSITY_INIT_MARKER in launcher_text:
        return launcher_text  # already wrapped
    fallback = (
        'echo "InterfaceForge: density initialization failed; VASP uses its standard start" >&2'
        if on_failure == "standard"
        else 'echo "InterfaceForge: density initialization failed; aborting job" >&2; exit 3'
    )
    if _PRECONDITION_MARKER in launcher_text:
        # The fresh-start SCF is the NSW=0 preconditioner; the MD restarts
        # from its WAVECAR, so the density seed belongs in precondition/.
        if on_failure == "abort":
            raise SafetyError("--density-init-on-failure abort is not supported with --precondition")
        if launcher_text.count(_PRECONDITION_STEP) != 1:
            raise SafetyError(f"{launcher_name}: cannot locate the preconditioner step to seed")
        inline = f"{{ {hook} || {fallback}; }} && "
        wrapped = launcher_text.replace(_PRECONDITION_STEP, _PRECONDITION_STEP + inline)
        return wrapped.replace(
            f"# --- {_PRECONDITION_MARKER}",
            f"# --- {DENSITY_INIT_MARKER}: neural initial density seeds the preconditioner SCF ---\n"
            f"# --- {_PRECONDITION_MARKER}",
            1,
        )

    lines = launcher_text.splitlines()
    hits = [i for i, line in enumerate(lines) if _VASP_INVOCATION.match(line)]
    if len(hits) != 1:
        raise SafetyError(
            f"--density-init needs {launcher_name} to have exactly one line that runs vasp; "
            f"found {len(hits)}. Add the initialize-density call by hand."
        )
    index = hits[0]
    vasp_line = lines[index]
    indent = vasp_line[: len(vasp_line) - len(vasp_line.lstrip())]
    block = [
        f"{indent}# --- {DENSITY_INIT_MARKER}: neural initial density before VASP ---",
        f"{indent}if ! {hook}; then",
        f"{indent}    {fallback}",
        f"{indent}fi",
        vasp_line,
    ]
    wrapped = lines[:index] + block + lines[index + 1 :]
    return "\n".join(wrapped) + ("\n" if launcher_text.endswith("\n") else "")
