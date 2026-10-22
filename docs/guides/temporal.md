# Time and forgetting

Every memory carries a timestamp `t` (default: now, in UTC) and optional `valid_from` / `valid_until`. A **temporal policy** turns those into a weight in \((0, 1]\) at query time; the policy is a constructor argument of `Memory`.

## Half-life

```python
from datetime import datetime, timedelta, timezone
from poincare_memory import HalfLife, Memory, generative_agents_recency

now = datetime.now(timezone.utc)
mem = Memory(decay=HalfLife(hours=24 * 14))
mem.add("Calf tightness after hill sprints", id="a", t=now - timedelta(days=28))
mem.add("Completed 6 x 300 m at 95 %, felt strong", id="b", t=now - timedelta(days=1))
for r in mem.search("training this week", k=2):
    print(r.id, round(r.time_weight, 3))
```

A memory 28 days old under a 14-day half-life has weight 0.25. `generative_agents_recency()` is the \(0.995^{\text{hours}}\) recency of Generative Agents expressed as a half-life of about 138 hours.

## Ebbinghaus with access strengthening

```python
from poincare_memory import Ebbinghaus

mem = Memory(decay=Ebbinghaus(stability_hours=24 * 7, boost_per_access=0.5))
```

Weight is \(e^{-\Delta t / S}\) with \(S = \text{stability\_hours}\cdot(1 + \text{boost\_per\_access}\cdot\text{accesses})\). Every time `search` returns a memory it increments the memory's `access_count` (disable with `touch=False`), so frequently retrieved memories decay more slowly.

## Validity windows for facts that stop being true

```python
from poincare_memory import Product, ValidityWindow

mem = Memory(decay=Product(HalfLife(hours=24 * 90), ValidityWindow()))
mem.add(
    "API rate limit is 100 rps (release 2.3)",
    id="old",
    t=now - timedelta(days=90),
    valid_until=now - timedelta(days=20),
)
mem.add(
    "API rate limit is 250 rps (release 2.4)",
    id="new",
    t=now - timedelta(days=20),
    valid_from=now - timedelta(days=20),
)
print([r.id for r in mem.search("rate limit", k=2)])  # ['new', 'old']
print(
    [r.id for r in mem.search("rate limit", k=2, at_time=now - timedelta(days=40))]
)  # ['old', 'new']
```

Outside its window a memory keeps a tiny weight (`outside_weight`, default \(10^{-6}\)) rather than disappearing, so it can still be found when nothing else matches; filter it out explicitly if you want it gone (`filters=lambda r: ...`).

## How weight enters the score

`score = -distance + depth_bias * depth + time_coef * log(weight)`. With `time_coef=1`, a halving of the weight costs \(\ln 2 \approx 0.69\) score units, the same as a memory being 0.69 hyperbolic units further away. Set `time_coef` higher to let recency dominate, or 0 to ignore time while keeping the timestamps for inspection.

## Asking about the past

`search(at_time=some_datetime)` evaluates the policy at that moment, which is how an agent can reconstruct what it would have retrieved on a given day. The inspector's time slider does the same visually.
