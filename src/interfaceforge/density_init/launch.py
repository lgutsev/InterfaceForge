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
initializer failure is counted, never hidden), also under ``--precondition``.
Either way the job writes ``density_init_fallback.json`` recording the action.
"""

from __future__ import annotations

import shlex
import sys

from ..errors import SafetyError
from ..vasp import _PRECONDITION_MARKER, _VASP_INVOCATION

DENSITY_INIT_MARKER = "InterfaceForge --density-init"
ON_FAILURE = ("standard", "abort")
FALLBACK_NAME = "density_init_fallback.json"
FALLBACK_FORMAT = "interfaceforge-density-init-fallback"
ABORT_EXIT_CODE = 3
_PRECONDITION_STEP = "mv -f INCAR.precondition INCAR && "


def default_interfaceforge_command() -> str:
    return f"{shlex.quote(sys.executable)} -m interfaceforge"


def vasp_command_from_launcher(text: str) -> str | None:
    hits = [line.strip() for line in text.splitlines() if _VASP_INVOCATION.match(line)]
    return hits[-1] if hits else None


def _fallback_commands(on_failure: str) -> list[str]:
    """Shell run when the hook failed: record the fallback, then continue or abort.

    ``density_init_fallback.json`` is written next to the run's inputs so a
    status audit can tell *requested* from *executed* initialization without
    reading job logs; the hook block removes it first so a requeue that
    succeeds leaves no stale record.
    """

    message = (
        "density initialization failed; VASP uses its standard start"
        if on_failure == "standard"
        else "density initialization failed; aborting job"
    )
    commands = [
        "printf '{\"format\": \"%s\", \"action\": \"%s\", \"exit_code\": %s, \"at\": \"%s\"}\\n' "
        f"{FALLBACK_FORMAT} {on_failure} \"$density_init_rc\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" "
        f"> {FALLBACK_NAME}",
        f'echo "InterfaceForge: {message} (exit $density_init_rc)" >&2',
    ]
    if on_failure == "abort":
        commands.append(f"exit {ABORT_EXIT_CODE}")
    return commands


def wrap_launcher_with_density_init(
    launcher_text: str, *, launcher_name: str, hook: str, on_failure: str = "standard"
) -> str:
    """Insert ``hook`` so it runs right before the fresh-start VASP call.

    The hook is written ``hook || density_init_rc=$?`` so a launcher running
    under ``set -e`` still reaches the fallback branch.
    """

    if on_failure not in ON_FAILURE:
        raise SafetyError(f"--density-init-on-failure must be one of {', '.join(ON_FAILURE)}")
    if DENSITY_INIT_MARKER in launcher_text:
        return launcher_text  # already wrapped
    fallback = _fallback_commands(on_failure)
    if _PRECONDITION_MARKER in launcher_text:
        # The fresh-start SCF is the NSW=0 preconditioner; the MD restarts
        # from its WAVECAR, so the density seed belongs in precondition/.
        lines = launcher_text.splitlines()
        hits = [i for i, line in enumerate(lines) if _PRECONDITION_STEP in line]
        if len(hits) != 1:
            raise SafetyError(f"{launcher_name}: cannot locate the preconditioner step to seed")
        index = hits[0]
        step_line = lines[index]
        indent = step_line[: len(step_line) - len(step_line.lstrip())]
        inline = (
            f"{{ rm -f {FALLBACK_NAME}; density_init_rc=0; {hook} || density_init_rc=$?; "
            f'if [ "$density_init_rc" -ne 0 ]; then {"; ".join(fallback)}; fi; }} && '
        )
        block = [step_line.replace(_PRECONDITION_STEP, _PRECONDITION_STEP + inline, 1)]
        if on_failure == "abort":
            # ``exit`` above only leaves the preconditioner subshell; stop the job here.
            block.append(
                f"{indent}if grep -qs '\"action\": \"abort\"' precondition/{FALLBACK_NAME}; then "
                f'echo "InterfaceForge: density initialization failed; aborting job" >&2; '
                f"exit {ABORT_EXIT_CODE}; fi"
            )
        wrapped_lines = lines[:index] + block + lines[index + 1 :]
        wrapped = "\n".join(wrapped_lines) + ("\n" if launcher_text.endswith("\n") else "")
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
        f"{indent}rm -f {FALLBACK_NAME}",
        f"{indent}density_init_rc=0",
        f"{indent}{hook} || density_init_rc=$?",
        f'{indent}if [ "$density_init_rc" -ne 0 ]; then',
        *(f"{indent}    {command}" for command in fallback),
        f"{indent}fi",
        vasp_line,
    ]
    wrapped = lines[:index] + block + lines[index + 1 :]
    return "\n".join(wrapped) + ("\n" if launcher_text.endswith("\n") else "")
