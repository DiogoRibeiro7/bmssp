# API stability policy

`ssspx` follows [semantic versioning](https://semver.org/) with a major version of
`0`, meaning that public APIs may change at each minor release.  To minimize
surprises, we document which modules and classes are considered public and
covered by our compatibility guarantees.

## Public surface

The following modules are **public**:

- `ssspx.graph`
- `ssspx.graph_numpy`
- `ssspx.solver`
- `ssspx.transform`
- `ssspx.io`
- `ssspx.export`
- `ssspx.frontier`
- `ssspx.logger`
- `ssspx.exceptions`

Only objects re-exported from :mod:`ssspx.__init__` via ``__all__`` are part of
the stable API.  Everything else should be treated as private and may change
without notice.

## Deprecations

When public APIs change, the old names are kept for at least one minor release
and emit a :class:`DeprecationWarning`.  Warnings include the version when the
name was deprecated and the release in which it will be removed.  See
:func:`ssspx.deprecation.warn_once` for details.

Code that relies on deprecated names should migrate before the ``remove_in``
version.  Our CI checks fail if a deprecation past its ``remove_in`` version
remains in the codebase.
