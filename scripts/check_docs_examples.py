"""Execute every ```python block in docs/**/*.md so the documentation cannot drift from the code.

Blocks containing ``# docs: skip`` are not executed. Each file's blocks run in order in one
namespace inside a temporary working directory. Usage: python scripts/check_docs_examples.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def main() -> int:
    failures = 0
    files = sorted((ROOT / "docs").rglob("*.md"))
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for md in files:
            blocks = [
                b for b in BLOCK.findall(md.read_text(encoding="utf-8")) if "# docs: skip" not in b
            ]
            if not blocks:
                continue
            ns: dict[str, object] = {}
            for n, code in enumerate(blocks, 1):
                try:
                    exec(compile(code, f"{md.name}#{n}", "exec"), ns)  # intentional: docs examples
                except Exception:
                    failures += 1
                    print(f"FAIL {md.relative_to(ROOT)} block {n}:\n{traceback.format_exc()}")
            print(f"ok   {md.relative_to(ROOT)}: {len(blocks)} block(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
