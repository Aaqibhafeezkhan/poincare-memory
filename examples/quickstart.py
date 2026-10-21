"""Quickstart: build a small hierarchy with dated events, search it three ways, inspect it."""

from datetime import datetime, timedelta, timezone

from poincare_memory import HalfLife, Memory

now = datetime.now(timezone.utc)
mem = Memory(decay=HalfLife(hours=24 * 30))
body = mem.add("Human musculoskeletal system", id="body")
lower = mem.add("Lower limb musculature", parent=body, id="lower")
hams = mem.add("Hamstring group: knee flexion and hip extension", parent=lower, id="hams")
bf = mem.add("Biceps femoris: lateral hamstring", parent=hams, id="bf")
mem.add(
    "Athlete A reported tight hamstrings after 400 m repeats",
    parent=hams,
    t=now - timedelta(days=30),
)
mem.add(
    "Athlete A: flexibility improved after two weeks of eccentric work",
    parent=hams,
    t=now - timedelta(days=3),
)

print("nearest:")
for r in mem.search("hamstring tightness", k=3):
    print(f"  {r.score:6.2f}  depth={r.depth:.1f}  w={r.time_weight:.2f}  {r.text}")
print("\nsubtree of 'lower':")
for r in mem.search("knee", mode="subtree", root=lower, k=3):
    print(f"  {r.text}")
print("\nancestors of 'bf':", [r.id for r in mem.search(mode="ancestors", root=bf)])
print("\n" + mem.tree())
mem.inspect("quickstart.html")
print("\nwrote quickstart.html")
