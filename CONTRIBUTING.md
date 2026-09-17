# Contributing

Install with `python -m pip install -e ".[dev,reproduce]"`, then run
`python -m pytest tests` before submitting a pull request. Keep model changes,
data preparation and evaluation changes explicit, with suitable tests.

Python implementations belong in `src/motrust/`; protocol resources belong
in `src/motrust/resources/`. Keep downloads, fitted models and generated
outputs in the selected work directory, outside the installed package.
Do not commit credentials, matrices, checkpoints or generated results.

The source manifest identifies the distributed code. After an intentional
reviewed change, regenerate it with `python scripts/update_source_manifest.py`.
