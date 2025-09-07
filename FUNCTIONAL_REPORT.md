# Functional Readiness Report

## 1. Setup
- `poetry install -E numpy-backend`
- `pre-commit install`
- Python: 3.12.10
- Poetry: 2.1.4
- Key packages: bandit 1.8.6, hypothesis 6.138.8, interrogate 1.7.0, mkdocs 1.6.1, mkdocs-material 9.6.18, mkdocstrings 0.30.0, mypy 1.17.1, numpy 2.0.2, pydocstyle 6.3.0, pytest 8.4.1, python-semantic-release 9.21.1

## 2. Library smoke
- Solved sample graph: distances `[0.0, 1.0, 3.0, 4.0]`, path to 3 `[0, 1, 2, 3]`
- Re-ran with `use_transform=False`, `frontier="heap"`, and NumPy backend – all consistent

## 3. CLI end-to-end
- `poetry run ssspx --edges docs/examples/small.csv --source 0 --target 3 --frontier block --export-json /tmp/sp.json --export-graphml /tmp/sp.graphml`
- `poetry run ssspx --random --n 100 --m 400 --source 0 --frontier heap --no-transform --profile --profile-out /tmp/prof.cprof`
- `/tmp/sp.json` parsed (4 nodes, 3 edges); `/tmp/sp.graphml` parsed; stdout distances `[0.0, 1.0, 3.0]`, path `[0, 1, 2, 3]`

## 4. Unit & property tests
- `poetry run pytest -q --maxfail=1 --disable-warnings --cov=ssspx --cov-report=term-missing`
- Coverage: 94% lines, 94% branches (threshold 95% lines)

## 5. Integration tests
- `poetry run pytest -q -m integration` → 11 passed

## 6. Regression tests
- `poetry run pytest -q -m regression` → 2 passed

## 7. Docstrings & docs build
- `poetry run pydocstyle src/ssspx --convention=google` → no issues
- `poetry run interrogate -c pyproject.toml` → 91.1% docstring coverage (>=90% target)
- `poetry run mkdocs build --strict` → site built cleanly

## 8. Packaging sanity
- `poetry build` → built `ssspx-0.1.0.tar.gz` and wheel
- `poetry run python -m twine check dist/*` → both artifacts passed; METADATA name/version/summary present

## 9. Security & hygiene
- `poetry run bandit -q -r src/ssspx -ll` → 0 HIGH, 1 MEDIUM, 4 LOW
- `poetry run pip-audit` → vulnerabilities: cryptography GHSA-79v4-65xg-pq4g (fix 44.0.1); py PYSEC-2022-42969

## 10. JOSS & Zenodo
- `paper/paper.md` and `paper.bib` present with required sections
- `cffconvert --validate --infile CITATION.cff` → missing (install via `pip install cffconvert`)
- `.zenodo.json` parsed; title/description/creator ORCID present

## 11. Bench run
- `PYTHONPATH=src poetry run python -m ssspx.bench` → fails (requires `--sizes`)
- `PYTHONPATH=src poetry run python -m ssspx.bench --trials 1 --sizes 10,20` → runs and outputs timing table

## 12. Cross-version smoke
- `pyenv versions` shows 3.10–3.13 but not 3.9; switching to 3.11.12 and reinstalling deps failed (`Cannot install six`) due to PEP 668 protections

---

### Summary
| Step | Result |
|------|--------|
| 1 | PASS |
| 2 | PASS |
| 3 | PASS |
| 4 | **FAIL** – coverage 94% (<95%) |
| 5 | PASS |
| 6 | PASS |
| 7 | PASS |
| 8 | PASS |
| 9 | **FAIL** – dependency vulnerabilities |
|10 | WARN – citation tool missing |
|11 | **FAIL** – bench requires `--sizes` |
|12 | **FAIL** – cross-version installs blocked / Python 3.9 absent |

### Remediation
1. Raise coverage ≥95% and ensure pytest exits cleanly.
2. Upgrade `cryptography` ≥44.0.1 and replace/remove `py` package.
3. Add default `--sizes` or guidance so `python -m ssspx.bench` works without arguments.
4. Add `cffconvert` to dev deps and CI for citation validation.
5. Enable virtualenvs or use `--break-system-packages` to test under alternate Python versions; install Python 3.9 if needed.
