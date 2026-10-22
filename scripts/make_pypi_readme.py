"""Derive README.pypi.md from README.md for the PyPI project page.

PyPI renders the long description without access to repository files, so image
lines and the CI badge are dropped and the inspector demo is described in words.
Run without arguments to regenerate; ``--check`` exits 1 if the file is stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "README.md"
DST = ROOT / "README.pypi.md"
IMAGE_LINE = re.compile(r"^!\[.*\]\(.*\)\s*$")


def render(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if IMAGE_LINE.match(line):
            if "inspector" in line.lower():
                out.append(
                    "*An animated demo of the interactive inspector (Poincaré disk, time slider, "
                    "table and tree views) and the benchmark chart are in the repository README.*"
                )
            continue
        if "actions/workflows/ci.yml/badge.svg" in line:
            # keep the registry, licence and paper badges; drop the repository-hosted CI badge
            line = re.sub(r"\[!\[CI\]\([^)]*\)\]\([^)]*\)\s*", "", line)
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    rendered = render(SRC.read_text(encoding="utf-8"))
    if "--check" in sys.argv:
        if not DST.exists() or DST.read_text(encoding="utf-8") != rendered:
            print("README.pypi.md is stale: run python scripts/make_pypi_readme.py")
            return 1
        print("README.pypi.md is up to date")
        return 0
    DST.write_text(rendered, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
