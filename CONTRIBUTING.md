# Contributing

Keep engine-specific assumptions explicit and test generated shell with
`bash -n`. Do not commit proprietary VASP files, scientific campaign data,
cluster credentials, large checkpoints or container images.

Run:

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

Bug reports should include the InterfaceForge version, relevant generated
manifest, engine version and a sanitized minimal input. Never attach POTCAR.
