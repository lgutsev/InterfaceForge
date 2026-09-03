# Archiving and Hugging Face packaging

> **Verification note:** automated file-level coverage only. Collection and
> packaging verify membership, distinct content, checksums and archive
> structure. No real Hugging Face upload, and no archive→restore→retrain cycle,
> has been exercised end to end.

InterfaceForge's standardized outputs — four-member MACE and DeePMD committees
and the canonical `iface collect` dataset — can be turned into two kinds of
portable, checksummed artifact:

- a **dataset archive**: one `.zip` of a canonical dataset for a backup drive,
  restored with `unzip`;
- a **Hugging Face model repository**: an upload-ready directory with a generated
  model card, `.gitattributes` for Git LFS, a provenance manifest and the exact
  `hf upload` command.

**InterfaceForge never contacts the Hugging Face Hub.** `iface package
huggingface` stops at a ready-to-push directory; you run `hf upload` yourself.
`huggingface_hub` is not a dependency.

## Dataset archive (cold storage / reuse)

```bash
# During collection:
iface collect -c campaign.yaml --archive backups/sintin_dataset_v1.zip

# Or later, against any collected dataset directory:
iface package dataset-archive datasets/canonical backups/sintin_dataset_v1.zip \
    --label "SiN/TiN dataset v1"
```

The archive contains one top-level directory:

```text
sintin_dataset_v1/
  README.md                      # what it is, restore + verify recipe
  interfaceforge_manifest.json   # the collect manifest + archive metadata
  checksums.sha256
  data/
    manifest.json manifest.csv frames.csv
    train.extxyz valid.extxyz test.extxyz
    deepmd/{train,valid,test}/<system>/...
```

`data/` is byte-for-byte a canonical dataset directory, so restore is just
`unzip`; repoint the campaign's MACE `train_file` / `valid_file` / `test_file`
and DeePMD `dataset_root` at the restored `data/` and run `iface train`.

Options: `--no-extxyz` keeps only the DeePMD NPY tree; `--compression stored`
skips compression for very large datasets; `--force` overwrites an existing
archive. POTCAR and OUTCAR files are never included.

## Collecting a DeePMD committee

`iface committee collect` now takes `--engine deepmd`. Point it at a
`models/deepmd/<arch>` directory containing the trained `model_NNN/` runs; each
member must have a non-empty frozen model (`frozen_model.pth` / `.pb`), so run
`dp freeze` (or the generated `run_ensemble.slurm`, which freezes automatically)
first.

```bash
iface committee collect models/deepmd/dpa2 stored_models/sintin_dpa2_v1 \
    --engine deepmd --expected-members 4 --label "SiN/TiN DPA-2 v1"
iface committee verify stored_models/sintin_dpa2_v1
```

The bundle has the same shape as a MACE bundle — `models/model_000.pth`, …,
`manifest.json`, `checksums.sha256`, `committee-models.txt`, `README.md`, and a
`.zip` — plus DeePMD provenance in the manifest (`architecture`, `backend`,
`type_map`, `numb_steps`, `base_checkpoint` for `*_ft`, and each member's parsed
`input.json`). Seeds are recovered from the sibling `ensemble_manifest.json`
when it is present.

## Hugging Face model repository

```bash
iface package huggingface stored_models/sintin_dpa2_v1 hf/sintin_dpa2_v1 \
    --repo-id myorg/sintin-dpa2 \
    --license mit \
    --dataset-repo-id myorg/sintin-dft \
    --metrics models/deepmd/evaluation/dpa2/job_1234567/rmse_overall.csv \
    --zip
iface package verify hf/sintin_dpa2_v1
```

The input is an **extracted** committee bundle (MACE or DeePMD). The output is:

```text
hf/sintin_dpa2_v1/
  README.md                      # model card: YAML frontmatter + body
  .gitattributes                 # *.model *.pth *.pb ... through Git LFS
  interfaceforge_manifest.json   # engine, architecture, members, metrics, provenance
  checksums.sha256
  UPLOAD.md                      # the exact `hf upload` commands
  models/                        # copied verbatim from the bundle
    model_000.pth  model_000.input.json  ...
  committee-models.txt
```

The generated card fills in `library_name` (`mace` / `deepmd-kit`), `tags`,
`license`, `base_model` (fine-tunes), `datasets`, and a `model-index` block when
metrics are supplied. The body has a committee-members table, training protocol,
data provenance, load snippets, an evaluation table, a scientific-maturity
caveat, and a citation block. It never claims the potential is validated or
production-ready — replace anything marked `TODO` before uploading.

### Metrics

Pass `--metrics` a `comparison.json` from `iface mlip-compare finalize` or a
DeePMD `rmse_overall.csv`. For DeePMD, if `--metrics` is omitted the packager
also looks for the newest
`models/deepmd/evaluation/<arch>/job_*/rmse_overall.csv` next to the source
runs.

### Fine-tuned committees

For a `dpa2_ft` / `dpa3_ft` committee the card is tagged `fine-tuned` and names
the local foundation checkpoint. Pass `--base-model <hub-id>` to also set the
`base_model` frontmatter field to the checkpoint's public Hub id.

## Verifying

`iface package verify <path>` accepts a directory or a `.zip` and dispatches on
the recorded `artifact_type`: it recomputes every `checksums.sha256` entry for
dataset archives and Hugging Face packages, and delegates committee bundles /
training-data archives to `iface committee verify`.

```bash
iface package verify backups/sintin_dataset_v1.zip
iface package verify hf/sintin_dpa2_v1
# from an extracted top directory:
sha256sum -c checksums.sha256
```

## Uploading (you do this)

```bash
pip install -U huggingface_hub
hf auth login
hf repo create myorg/sintin-dpa2 --repo-type model
hf upload myorg/sintin-dpa2 hf/sintin_dpa2_v1 --repo-type model
```

`UPLOAD.md` inside the package repeats these with your repo id filled in.
