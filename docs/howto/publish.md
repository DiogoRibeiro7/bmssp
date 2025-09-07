# Publishing

## PyPI

Tagged releases trigger CI to build source and binary wheels. To publish manually:

```bash
poetry build
poetry run twine check dist/*
poetry publish # or twine upload dist/*
```

Semantic-release automates version bumps and GitHub Releases. Configure a PyPI token in the
`PYPI_TOKEN` secret and enable the release workflow.

## conda-forge

A starter recipe lives under `conda/meta.yaml`. After a PyPI release:

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes).
2. Copy `conda/meta.yaml` into `recipes/ssspx/meta.yaml` and submit a pull request.
3. Once merged, a dedicated `ssspx-feedstock` repository will be created where future
   updates are managed via version bumps in the recipe.

Refer to the [conda-forge docs](https://conda-forge.org/docs/) for detailed instructions
and review guidelines.
