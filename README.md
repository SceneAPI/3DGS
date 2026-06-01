# 3DGS

3DGS standard plugins.

This repository is currently a minimal Python package scaffold prepared for PyPI publishing.

## Installation

```bash
pip install 3DGS
```

## Publishing

Publishing is handled by `.github/workflows/publish.yml` through PyPI Trusted Publishing.

Configure the PyPI trusted publisher with:

- Owner: `SceneAPI`
- Repository: `3DGS`
- Workflow name: `publish.yml`
- Environment name: `pypi`

Then publish a GitHub release for a version that has not already been uploaded to PyPI.
