"""An agent loop that writes observations into memory and retrieves time-weighted context.

No LLM is needed to run this; replace ``answer`` with a model call to turn it into an agent.
"""

from datetime import datetime, timedelta, timezone

from poincare_memory import Ebbinghaus, Memory

mem = Memory(decay=Ebbinghaus(stability_hours=24 * 7, boost_per_access=0.5))
now = datetime.now(timezone.utc)
athlete = mem.add("Athlete profile: 400 m sprinter, 24, right-leg dominant", id="athlete")
mem.add("Injury history", parent=athlete, id="injury")
mem.add("Training log", parent=athlete, id="log")
mem.add("Grade 1 hamstring strain, left, 2025-11", parent="injury", t=now - timedelta(days=320))
mem.add("Calf tightness after hill sprints", parent="log", t=now - timedelta(days=12))
mem.add("Completed 6 x 300 m at 95 %, felt strong", parent="log", t=now - timedelta(days=2))


def context(question: str, k: int = 3) -> str:
    hits = mem.search(question, k=k)  # accesses strengthen Ebbinghaus stability
    return "\n".join(f"- ({h.time_weight:.2f}) {h.text}" for h in hits)


def answer(question: str) -> str:
    return f"Q: {question}\nContext:\n{context(question)}\nA: <model answer here>"


print(answer("how is the athlete's hamstring?"))
print(answer("what happened in training this week?"))
