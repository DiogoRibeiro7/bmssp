# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-13

### Removed

- **BREAKING:** Removed the deprecated `load_graph` alias (deprecated since
  0.1.0). Use `read_graph` instead.

### Changed

- Updated development dependencies: mypy 2.2.0, pytest 9.1.1, hypothesis
  6.156.1, pip-audit 2.10.1, python-semantic-release 10.6.1, cryptography
  49.0.0, nbclient 0.11.0, soupsieve 2.8.4, and the pip group (bleach, idna,
  mistune, msgpack, nbconvert, pip, pymdown-extensions, tornado, urllib3).

### Security

- Bumped transitive dev dependencies GitPython to 3.1.51 and Pygments to
  2.20.0 to resolve Dependabot advisories (5 high, 1 low).

[1.1.0]: https://github.com/DiogoRibeiro7/bmssp/compare/v1.0.0...v1.1.0
