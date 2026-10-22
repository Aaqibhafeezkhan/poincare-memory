# Using it in an agent

A memory layer has two moments: write and read.

```python
from datetime import datetime, timezone
from poincare_memory import Ebbinghaus, Memory

mem = Memory(decay=Ebbinghaus(stability_hours=24 * 7, boost_per_access=0.5))
athlete = mem.add("Athlete profile: 400 m sprinter, 24, right-leg dominant", id="athlete")
mem.add("Injury history", parent=athlete, id="injury")
mem.add("Training log", parent=athlete, id="log")


def observe(text: str, where: str) -> str:
    """Write: attach an observation under the right branch."""
    return mem.add(text, parent=where, t=datetime.now(timezone.utc))


def context(question: str, k: int = 4) -> str:
    """Read: time-weighted, hierarchy-aware context for the model."""
    hits = mem.search(question, k=k)  # accesses strengthen stability
    return "\n".join(f"- [{h.id}] ({h.time_weight:.2f}) {h.text}" for h in hits)


observe("Calf tightness after hill sprints", "log")
observe("Grade 1 hamstring strain, left, 2025-11", "injury")
print(context("how is the athlete's hamstring?"))
```

Patterns that work well:

- **Structural nodes are timeless, observations are dated.** Give branches no decay by dating them "now" or by filtering on `meta`; let observations carry real timestamps.
- **Route by branch.** When you know the topic, search `mode="subtree"` under it first and fall back to global search; the companion package `poincare-orchestra` does exactly this per sub-task.
- **Let validity windows retire facts** instead of deleting them, so the past stays reconstructible with `at_time`.
- **Inspect before you trust.** `mem.inspect()` shows what the agent knows; `mem.stats()` shows how old it is.
- **Persist per user or per session** with `save()`; reload with the same embedder.

The orchestration side of the paper, with a metacognitive controller, provider-agnostic ensemble and multi-objective synthesis, is implemented in [`poincare-orchestra`](https://github.com/dhruvmojila/poincare-orchestra).
