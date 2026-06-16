---
name: setup
description: Use when setting up or validating the local spatialized development environment, installing extras, running tests, checking CLI availability, or preparing package build artifacts.
---

# Setup

Use this skill for repository setup, dependency installation, validation, and
release-readiness checks. Do not use it for ordinary spatialized modelling
workflow steps; use `spatialized-workflow` for those.

## Development Environment

Install all development extras:

```bash
.venv/bin/pip install -e '.[dev]'
```

The `dev` extra includes test, model, raster, and release tooling.

## Validation

For any code or workflow change, run:

```bash
.venv/bin/pytest -q
```

For CLI changes, smoke-test at least one installed command:

```bash
.venv/bin/spatialized feature-layout --layer mag:3 --rotations
```

For packaging or release-related changes, also run:

```bash
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

## Expected State

- Tests pass.
- CLI commands resolve from `.venv/bin/spatialized`.
- Build artifacts pass `twine check`.
- Generated artifacts in `dist/` remain untracked.
