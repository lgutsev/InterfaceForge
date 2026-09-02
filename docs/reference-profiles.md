# Literature reference profiles

A *reference profile* is a small YAML file describing one published paper's
reported interface quantities (work of adhesion, surface energy, ...) and the
DFT protocol that produced them. A campaign uses it to check its own computed
values against the literature automatically.

## Using a bundled profile

```bash
iface reference list                 # names of the bundled profiles
iface reference show sharifi2026     # the profile + the validation.references it expands to
```

Opt a campaign in — either add the line by hand:

```yaml
validation:
  reference_profiles: [sharifi2026]
```

or let InterfaceForge add it:

```bash
iface reference activate sharifi2026 -c campaign.yaml           # dry run: prints the result
iface reference activate sharifi2026 -c campaign.yaml --write   # applies it
```

`activate` makes the smallest possible edit to `campaign.yaml` — it splices the
name into `validation.reference_profiles` (creating the block or the list if
needed) and leaves every comment and the rest of the file untouched. It refuses
rather than guess when the edit is ambiguous (an inline `validation: {...}`, a
`reference_profiles` that is not a list, a list that spans lines), and it never
writes a result that fails to re-load. `--write` is required to change the file;
without it the proposed file is printed and nothing is touched.

`load_campaign` expands each named profile into `validation.references` entries
(one per quantity), copying `key`, `doi`, `citation` and `method` onto each so
every entry is self-describing. Hand-written `validation.references` entries are
still honoured and **win** over a profile entry with the same `(key, quantity)`
— handy for overriding a single value without editing the bundled file.

A profile name resolves against `interfaceforge/references/`; a path
(`./my_paper.yaml`) is loaded as-is.

## What consumes them

- `iface validate interface-energy` — `quantity: interface_energy` (none bundled
  yet; the excess-energy definition rarely appears in papers directly).
- `iface vasp adhesion audit -c campaign.yaml --interface <leaf>` and
  `iface validate adhesion -c campaign.yaml` — `quantity: work_of_adhesion`.

A computed value is compared to every `values` entry whose `match` keys all
appear (case-insensitively) in the interface metadata (`validation.interfaces`)
or, for the CSV path, in the row's own columns. The report gets the reference
value, the delta, and a within-`tolerance_j_per_m2` flag.

## Writing a profile

```yaml
schema_version: 1
key: myref2027
doi: "10.0000/xyz"
citation: "A. Author et al., Journal 1 (2027) 1."
method: {code: vasp, functional: PBE, encut_ev: 500, dispersion: d3}
references:
  - quantity: work_of_adhesion
    tolerance_j_per_m2: 0.4
    values:
      - {match: {orientation: "A(001)/B(111)", termination: N}, value_j_per_m2: 2.1}
```

`quantity` must be one of `work_of_adhesion`, `interface_energy`,
`surface_energy`. Drop the file in `src/interfaceforge/references/` to bundle
it, or keep it beside the campaign and reference it by path.

> **Verification:** bundled numbers are transcribed from paper tables by hand
> and have not been independently re-derived. See `docs/verification.md`.
