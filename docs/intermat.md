# InterMat adapter

> **Verification note:** code-only. No real InterMat generation campaign has
> yet been run and its structures reviewed by a human through this adapter.

InterfaceForge uses InterMat only as an optional crystalline-interface geometry
generator. InterMat performs surface construction, Zur lattice matching, and
film/substrate separation and lateral-registry scans. InterfaceForge retains
responsibility for VASP inputs, POTCARs, schedulers, reference labels, MLIP
training, validation, and work-function analysis.

This boundary is intentional. InterMat's calculator defaults are not imported,
and generated candidates never mutate `campaign.yaml` automatically.

## Install and inspect

```bash
python -m pip install -e '.[intermat]'
iface intermat status
```

The extra installs `spglib` explicitly because current InterMat/JARVIS surface
generation imports it even though the published dependency chain may omit it.

The adapter accepts **bulk crystalline structures in POSCAR format**, not
prebuilt slabs and not isolated molecular adsorbates.

## Generate candidates

```bash
iface intermat generate film_bulk.vasp substrate_bulk.vasp generated/intermat \
  --film-miller 0 0 1 \
  --substrate-miller 0 0 1 \
  --film-thickness 16 \
  --substrate-thickness 20 \
  --separation 2.5 \
  --separation 3.0 \
  --displacement-interval 0.25 \
  --max-area 300
```

The dedicated output directory contains:

- `structures/interface_####.vasp`: unique unrelaxed candidates;
- `intermat_manifest.json`: input hashes, parameters, mismatches, registry
  coordinates, candidate hashes, and provenance;
- `campaign_fragment.yaml`: `systems:` entries that can be reviewed and copied
  into an InterfaceForge campaign.

Structure paths in the fragment are relative to the fragment directory. Adjust
the prefix if the entries are copied into a campaign file elsewhere.

InterMat's inclusive displacement grid contains periodic duplicates at the
fractional 1.0 boundary. The adapter removes structurally identical candidates
before writing files. `--max-candidates` guards accidental combinatorial scans.

## Scientific checks before VASP

Every generated structure is only a geometric candidate. Inspect:

- surface termination and polarity;
- which constituent absorbed the lattice strain;
- interface stoichiometry and possible atom overlap;
- slab thickness and the existence of bulk-like internal layers;
- magnetic ordering and initial moments;
- vacuum and dipole-correction requirements.

Use `iface vasp geom inspect` and `iface vasp geom clean` as preliminary checks,
then prepare calculations using InterfaceForge's normal VASP workflow.

The adapter is intended for crystalline perovskite/transport-layer and other
solid/solid interfaces. Molecule-on-surface passivation remains an adsorption
workflow and is deliberately outside this adapter.
