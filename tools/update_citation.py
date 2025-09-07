"""Update project citation metadata with a real DOI.

Run this after publishing a release and obtaining a Zenodo DOI:

    python tools/update_citation.py 10.5281/zenodo.1234567
"""

from __future__ import annotations

import argparse
from pathlib import Path

PLACEHOLDER = "10.5281/zenodo.0000000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch DOI in citation files")
    parser.add_argument("doi", help="Zenodo DOI, e.g. 10.5281/zenodo.1234567")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    targets = [repo_root / "CITATION.cff", repo_root / "README.md"]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(PLACEHOLDER, args.doi), encoding="utf-8")
        print(f"Updated {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
