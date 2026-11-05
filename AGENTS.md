# Notes for agents

## Agentic assignments

If you were given an InterfaceForge assignment directory:

1. Read its rendered `ASSIGNMENT.md` (never work from `assignment.yaml` alone)
   and any dated entries in `NOTES.md`. The criteria are frozen: do not edit
   `assignment.yaml`, `ASSIGNMENT.md` or `frozen.json`.
2. Record `iface assignment git-snapshot <target repo>` before you start and
   again when you finish.
3. Return `report.json` matching
   [`schemas/assignment_report.schema.json`](schemas/assignment_report.schema.json):
   one verdict per task from that task's vocabulary, every value with a
   `source`, `not_assessed` and `deviations` always present.
4. Run `iface assignment validate <assignment dir> --report report.json` and
   include its output. Do not claim done while it lists failures other than
   tasks the assignment explicitly leaves to the PI.

Background: [docs/agentic-assignments.md](docs/agentic-assignments.md).
