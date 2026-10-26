"""Command-line interface: ``poincare-memory inspect|tree|search|demo``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="poincare-memory",
        description="Hierarchy-aware, time-aware memory on the Poincaré ball.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="write an interactive HTML inspector for a saved memory")
    s.add_argument("db", type=Path)
    s.add_argument("-o", "--out", type=Path, default=None, help="output .html (default: <db>.html)")
    s.add_argument("--open", action="store_true", help="open in the default browser")

    s = sub.add_parser("tree", help="print the hierarchy as an ASCII tree")
    s.add_argument("db", type=Path)
    s.add_argument("--root", default=None, help="print only the subtree below this memory ID")

    s = sub.add_parser("search", help="search a saved memory")
    s.add_argument("db", type=Path)
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", default="nearest")
    s.add_argument("--root", default=None)

    s = sub.add_parser("stats", help="print summary statistics of a saved memory as JSON")
    s.add_argument("db", type=Path)

    s = sub.add_parser("export", help="export a saved memory to portable JSON")
    s.add_argument("db", type=Path)
    s.add_argument("out", type=Path)

    s = sub.add_parser("import", help="import a JSON export into a new SQLite file")
    s.add_argument("json", type=Path)
    s.add_argument("db", type=Path)

    s = sub.add_parser("demo", help="build the anatomy demo, save it, and write its inspector")
    s.add_argument("-o", "--out", type=Path, default=Path("poincare-demo"), help="output prefix")
    s.add_argument("--open", action="store_true")

    a = p.parse_args(argv)
    from .store.memory import Memory

    if a.cmd == "inspect":
        mem = Memory.load(a.db)
        out = a.out or a.db.with_suffix(".html")
        mem.inspect(out, open_browser=a.open)
        print(f"wrote {out} ({len(mem)} memories)")
    elif a.cmd == "tree":
        print(Memory.load(a.db).tree(root=a.root))
    elif a.cmd == "search":
        mem = Memory.load(a.db)
        for r in mem.search(a.query, k=a.k, mode=a.mode, root=a.root, touch=False):
            print(
                f"{r.score:8.3f}  d={r.distance:5.2f}  w={r.time_weight:5.3f}  depth={r.depth:4.2f}  {r.record.id}  {r.record.text}"
            )
    elif a.cmd == "stats":
        import json

        print(json.dumps(Memory.load(a.db).stats(), indent=2))
    elif a.cmd == "export":
        Memory.load(a.db).to_json(a.out, indent=1)
        print(f"wrote {a.out}")
    elif a.cmd == "import":
        mem = Memory.from_json(a.json)
        mem.save(a.db)
        print(f"wrote {a.db} ({len(mem)} memories)")
    elif a.cmd == "demo":
        from .demo import build_demo_memory
        from .temporal import HalfLife

        mem = build_demo_memory(decay=HalfLife(hours=24 * 30))
        db = a.out.with_suffix(".db")
        html = a.out.with_suffix(".html")
        mem.save(db)
        mem.inspect(html, open_browser=a.open, title="poincare-memory · anatomy demo")
        print(mem.tree())
        print(f"\nsaved {db} and {html}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
