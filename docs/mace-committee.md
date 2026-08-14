# MACE committee bundles

> **Verification note:** automated file-level coverage only. Collection verifies
> membership, distinct content and storage integrity; it does not demonstrate
> that MACE can load the models or that their predictions are scientifically valid.

Completed committee models are useful independently of the training directories.
InterfaceForge can collect one final model from each `seed_*` run into a compact,
immutable deployment bundle without copying checkpoints, optimizer state, training
logs or the usually much larger extxyz dataset.

For the TiN/SiN layout:

```text
mace_committee/
  seed_0/mace_model/TiN_SiN_mace_stagetwo.model
  seed_211/mace_model/TiN_SiN_mace_stagetwo.model
  seed_307/mace_model/TiN_SiN_mace_stagetwo.model
  seed_419/mace_model/TiN_SiN_mace_stagetwo.model
```

create a versioned bundle with:

```bash
iface committee collect \
  /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/mace_committee \
  /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/stored_models/tin_sin_mace_v1.zip \
  --expected-members 4 \
  --label "TiN/SiN MACE committee v1"
```

The default pattern is `seed_*/mace_model/*_stagetwo.model`. Use
`--model-pattern` only when the final MACE filenames differ. Collection fails if
the expected number of models is absent, one run has multiple matching final
models, a model is empty, or two members have identical content.

The output contains:

```text
stored_models/tin_sin_mace_v1/
  models/seed_0.model
  models/seed_211.model
  models/seed_307.model
  models/seed_419.model
  committee-models.txt
  checksums.sha256
  manifest.json
  README.md
stored_models/tin_sin_mace_v1.zip
```

The output argument may be written either with or without `.zip`. Collection
always creates both the extracted directory, which AI2-Kit can use directly,
and a ZIP archive with one top-level directory for storage or transfer.

Training data are never placed in the committee directory or committee ZIP. To
archive them separately, add both options:

```bash
iface committee collect \
  /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/mace_committee \
  /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/stored_models/tin_sin_mace_v1.zip \
  --training-data /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/train.extxyz \
  --training-data-output /ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/stored_data/tin_sin_training_v1.zip
```

The second ZIP has its own manifest and checksums and records the associated
committee digest. Omit both training-data options when collecting only models.
For very large datasets, `--training-data-compression stored` avoids compression
CPU time but produces a larger archive.

`manifest.json` records the original run and model paths, seed identifiers, file
sizes, SHA-256 checksums, presence of the standard run-artifact directories and
optional training-data hashes. The training data are hashed in place rather than
copied into the bundle.

Bundles are intentionally immutable: collection refuses to replace an existing
output directory. Use a new versioned name when a committee changes. Verify a
stored or transferred bundle with:

```bash
iface committee verify stored_models/tin_sin_mace_v1
iface committee verify stored_models/tin_sin_mace_v1.zip
iface committee verify stored_data/tin_sin_training_v1.zip
cd stored_models/tin_sin_mace_v1
sha256sum -c checksums.sha256
```

For AI2-Kit, use the four files under the bundle's `models/` directory as
`active_learning.ai2kit.committee_models`. The original full training runs should
still be retained separately if future restart or forensic inspection is needed.
