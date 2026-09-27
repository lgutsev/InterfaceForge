# Neural density initialization examples

| File | Purpose |
|---|---|
| [`pilot.yaml`](pilot.yaml) | Five-case pilot for `iface vasp density-init-bench prepare` (2 nonmagnetic perovskite/interface, 1 metallic/narrow-gap interface, 2 NiO AFM-II) |
| [`create_ndi_env.sh`](create_ndi_env.sh) | Creates the isolated `neural_paw_dft` environment (pinned upstream commit) |
| [`example_density_init_nio_afm2.json`](example_density_init_nio_afm2.json) | `density_init.json` provenance for a NiO AFM-II run — **mocked backend**, illustrative schema only |
| [`example_bench_report.md`](example_bench_report.md) / [`.json`](example_bench_report.json) | `density-init-bench compare` output from synthetic OUTCAR fixtures — **not** a performance result |

See [`docs/density-init.md`](../../docs/density-init.md).
