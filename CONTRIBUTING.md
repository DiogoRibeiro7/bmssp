# Contributing

Thanks for your interest in contributing to ssspx!

## Requirements & Setup

- Python 3.9+
- Install dependencies and optional Git hooks:

```bash
poetry install
pre-commit install
```

Poetry uses isolated virtual environments. If your global configuration
disables them, re-enable with `poetry config virtualenvs.create true` or pass
`--break-system-packages` when installing. For cross-version testing (e.g.,
Python 3.9), ensure the interpreter is installed (`apt-get install python3.9`
on Debian/Ubuntu) and run `poetry env use 3.9`.

## Testing

Run the default test suite (skips integration and regression markers):

```bash
poetry run pytest -q
```

### Integration tests

```bash
poetry run pytest -q -m integration
```

### Regression snapshots

```bash
poetry run pytest -q -m regression
```

If a legitimate change alters solver output, refresh the stored snapshots:

```bash
poetry run python tools/update_snapshots.py
```

## Docstring style

Use Google-style docstrings:

```python
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Sum of a and b.

    Examples:
        >>> add(1, 2)
        3
    """
    return a + b
```

## Commit & branching

- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (`feat:`, `fix:`, `docs:`, etc.)
- Create feature branches from `main` and open pull requests
- Keep commits focused; rebase locally to maintain a linear history

## Dependency policy

Project dependencies specify minimum supported versions in `pyproject.toml`.
Upper bounds are avoided unless a newer major release breaks compatibility.
A scheduled workflow runs weekly to update the lockfile and pre-commit hooks.
If a dependency must be capped, note the reason in the associated pull request
and remove the cap as soon as upstream fixes are available.

## Snapshot workflow

Snapshot fixtures live under `tests/regressions/fixtures/` and record graph
parameters plus expected distances. Review the diff whenever updating snapshots
and commit the JSON files alongside code changes.

## Triage policy and SLA

Maintainers review new issues and pull requests within **five business days**.
During triage we apply labels, request missing details, and determine priority.
Critical security reports are addressed as soon as possible. If you have not
received a response after a week, feel free to leave a polite ping.

## Code of Conduct

Please read and abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
