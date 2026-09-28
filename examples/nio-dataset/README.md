# NiO canonical dataset and three-backend readiness

Reviewable settings for building **one** canonical NiO (+ phosphonate) dataset
and split that MACE, DeePMD/DPA and NequIP all consume. See
[docs/nio-dataset.md](../../docs/nio-dataset.md) and
[docs/nequip.md](../../docs/nequip.md).

| File | Purpose |
|---|---|
| `nio_dataset.yaml` | Export settings written out explicitly (all InterfaceForge defaults). |
| `campaign_models_snippet.yaml` | `models:` block pointing MACE, DeePMD and NequIP at the same canonical files. `models.nequip.r_max` is deliberately `null`: set it before training. |

## On LONI

Run on a compute/interactive node (parsing hundreds of OUTCARs is I/O heavy):

```bash
iface dataset readiness /path/to/NiO_head --config examples/nio-dataset/nio_dataset.yaml \
    --output audit/nio_readiness
less audit/nio_readiness/readiness.md          # machine-readable: readiness.json

iface dataset export /path/to/NiO_head --config examples/nio-dataset/nio_dataset.yaml \
    --output datasets/canonical
iface dataset verify datasets/canonical
iface dataset readiness /path/to/NiO_head --config examples/nio-dataset/nio_dataset.yaml \
    --output audit/nio_readiness --dataset datasets/canonical -c campaign.yaml
```

The second readiness run must report `ready_for_gpu_smoke_tests: true` before
any GPU time is spent; its blocking items say why otherwise.

## Synthetic demonstration (no cluster data needed)

The test fixture builds a small tree with the same `Step1/` → `Step2_<T>K/`
layout, including a truncated run, SCF-ceiling steps and a temperature
runaway, so the audit output can be inspected locally:

```bash
python - <<'PY'
import sys; from pathlib import Path
sys.path.insert(0, "tests")
from nio_fixture import build_nio_tree
build_nio_tree(Path("demo/NiO_head"), special={
    "Step2_600K/NiO_m110_Big_U46_OH75_scattered_capped": {"truncate_last": True},
    "Step2_450K/NiO_m110_Big_U46_OH50_clustered_dissoc": {"scf_ceiling_steps": (3,)},
    "Step2_600K/NiO_m110_Big_U46_OH50_clustered_dissoc": {"runaway_from": 5},
})
PY
iface dataset readiness demo/NiO_head --output demo/readiness
```

Numbers from the synthetic tree say nothing about the real campaign.
