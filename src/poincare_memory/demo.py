"""A small anatomy-flavoured demo hierarchy used by the CLI, docs and tests.

The content is generic textbook anatomy written for this project; it echoes the
paper's biomechanics setting (muscle groups → muscles → fibre types) without
using any dataset from the paper.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .store.memory import Memory

DEMO_TREE: list[tuple[str, str, str | None]] = [
    ("body", "Human musculoskeletal system", None),
    ("lower", "Lower limb musculature", "body"),
    ("upper", "Upper limb musculature", "body"),
    ("trunk", "Trunk and core musculature", "body"),
    ("quads", "Quadriceps femoris group: knee extension and hip flexion", "lower"),
    ("hams", "Hamstring group: knee flexion and hip extension", "lower"),
    ("glutes", "Gluteal group: hip extension, abduction and external rotation", "lower"),
    ("calf", "Triceps surae: plantar flexion of the ankle", "lower"),
    ("rf", "Rectus femoris: biarticular quadriceps head crossing hip and knee", "quads"),
    ("vl", "Vastus lateralis: largest quadriceps head, lateral thigh", "quads"),
    ("vm", "Vastus medialis: medial quadriceps head, terminal knee extension", "quads"),
    ("bf", "Biceps femoris: lateral hamstring, long and short heads", "hams"),
    ("st", "Semitendinosus: medial hamstring with a long distal tendon", "hams"),
    ("sm", "Semimembranosus: deep medial hamstring", "hams"),
    ("gmax", "Gluteus maximus: primary hip extensor in sprint acceleration", "glutes"),
    ("gmed", "Gluteus medius: pelvic stabiliser in single-leg stance", "glutes"),
    ("gastroc", "Gastrocnemius: biarticular, fast-twitch dominant plantar flexor", "calf"),
    ("soleus", "Soleus: slow-twitch dominant postural plantar flexor", "calf"),
    ("type2x", "Type IIx fibres: highest shortening velocity, fatigue quickly", "gastroc"),
    ("type1", "Type I fibres: fatigue resistant, oxidative metabolism", "soleus"),
    ("delts", "Deltoid group: shoulder abduction and flexion", "upper"),
    ("biceps", "Biceps brachii: elbow flexion and forearm supination", "upper"),
    ("lats", "Latissimus dorsi: shoulder extension and adduction", "upper"),
    ("abs", "Rectus abdominis: trunk flexion and intra-abdominal pressure", "trunk"),
    ("erector", "Erector spinae: spinal extension and posture", "trunk"),
    ("obl", "Obliques: trunk rotation and lateral flexion", "trunk"),
]

DEMO_EVENTS: list[tuple[str, str, int]] = [
    ("Athlete A reported tight hamstrings after 400 m repeats", "hams", 30),
    ("Athlete A: hamstring flexibility improved after two weeks of eccentric work", "hams", 10),
    ("Coach note: increase gluteus maximus activation drills before sprints", "gmax", 20),
    ("Athlete B: calf strain, soleus tender on palpation", "soleus", 45),
    ("Athlete B cleared to resume plyometrics", "soleus", 3),
    ("Team session focused on quadriceps strength at long muscle lengths", "quads", 60),
]


def build_demo_memory(**kwargs) -> Memory:  # type: ignore[no-untyped-def]
    """Return a populated demo :class:`Memory` (structural nodes plus dated events)."""
    now = datetime.now(timezone.utc)
    mem = Memory(**kwargs)
    for id_, text, parent in DEMO_TREE:
        mem.add(text, parent=parent, id=id_, t=now - timedelta(days=120))
    for text, parent, days_ago in DEMO_EVENTS:
        mem.add(text, parent=parent, t=now - timedelta(days=days_ago), meta={"kind": "event"})
    return mem
