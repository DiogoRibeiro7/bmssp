# Dependency policy

We pin minimum versions in `pyproject.toml` so that users know the oldest
supported release of each dependency. Upper version caps are avoided unless a
new major release is known to break compatibility.

Dependencies are refreshed automatically once a week. A scheduled workflow runs
`pre-commit autoupdate` and `poetry update` and opens a pull request labeled
`deps`. The full test suite and documentation build run on that pull request to
ensure updates do not break the project.

If a dependency introduces a breaking change, constrain the upper bound
minimally and file an issue to track removing the cap.
