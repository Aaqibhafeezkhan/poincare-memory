from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from poincare_memory import HalfLife, Memory, ValidityWindow, in_cone
from poincare_memory.demo import DEMO_TREE, build_demo_memory

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def fixed_clock():
    return T0


def test_add_search_and_structure():
    mem = Memory(clock=fixed_clock)
    body = mem.add("Human musculoskeletal system", id="body")
    lower = mem.add("Lower limb musculature", parent=body, id="lower")
    hams = mem.add("Hamstring group: knee flexion and hip extension", parent=lower, id="hams")
    bf = mem.add("Biceps femoris: lateral hamstring", parent=hams, id="bf")
    quads = mem.add("Quadriceps femoris group: knee extension", parent=lower, id="quads")
    assert len(mem) == 5
    assert mem.ancestors(bf) == [hams, lower, body]
    assert set(mem.descendants(lower)) == {hams, bf, quads}
    assert mem.siblings(hams) == [quads]
    assert mem.level(bf) == 3
    # depth grows by depth_step per level
    assert mem.depth(bf) == pytest.approx(mem.base_depth + 3 * mem.depth_step, abs=1e-6)
    res = mem.search("hamstring knee flexion", k=2)
    assert res[0].id in {hams, bf}
    assert mem.get(res[0].id).access_count == 1
    sub = mem.search("knee", mode="subtree", root=lower, k=10)
    assert {r.id for r in sub} <= {lower, hams, bf, quads}
    anc = mem.search(mode="ancestors", root=bf, k=10)
    assert [r.id for r in anc] and set(r.id for r in anc) == {hams, lower, body}
    assert (
        mem.search("hamstring knee flexion", k=1, metric="hyperbolic", touch=False)[0].id == body
    )  # generality bias


def test_cone_constrained_memory_makes_cone_queries_exact():
    mem = Memory(clock=fixed_clock, cone_constrained=True)
    r = mem.add("root", id="r")
    a = mem.add("branch a", parent=r, id="a")
    aa = mem.add("leaf under a", parent=a, id="aa")
    b = mem.add("branch b", parent=r, id="b")
    assert in_cone(mem.point(a), mem.point(aa), mem.ball, mem.cone_k, tol=1e-6)
    assert in_cone(
        mem.point(r), mem.point(aa), mem.ball, mem.cone_k, tol=1e-6
    )  # cones are transitive
    cone = mem.search(mode="cone", root=a, k=10)
    assert {x.id for x in cone} == {aa}
    assert all(x.cone_energy is not None and x.cone_energy <= 1e-9 for x in cone)
    assert b not in {x.id for x in mem.search(mode="cone", root=a, k=10)}


def test_time_weighting_changes_ranking():
    mem = Memory(decay=HalfLife(hours=24), clock=fixed_clock)
    old = mem.add("hamstring tightness reported", t=T0 - timedelta(days=10), id="old")
    new = mem.add("hamstring tightness reported", t=T0 - timedelta(hours=1), id="new")
    res = mem.search("hamstring tightness", k=2, touch=False)
    assert res[0].id == new and res[1].id == old
    assert res[0].time_weight > res[1].time_weight
    # evaluating at an earlier time restores parity
    res_then = mem.search("hamstring tightness", k=2, at_time=T0 - timedelta(days=11), touch=False)
    assert res_then[0].time_weight == pytest.approx(res_then[1].time_weight)


def test_validity_window_and_filters():
    mem = Memory(decay=ValidityWindow(), clock=fixed_clock)
    a = mem.add(
        "protocol v1 applies", valid_until=T0 - timedelta(days=1), id="a", meta={"kind": "rule"}
    )
    b = mem.add(
        "protocol v2 applies", valid_from=T0 - timedelta(days=1), id="b", meta={"kind": "rule"}
    )
    c = mem.add("unrelated note", id="c", meta={"kind": "note"})
    res = mem.search("protocol applies", k=3, touch=False)
    assert res[0].id == b
    rules = mem.search("protocol", k=5, filters={"kind": "rule"}, touch=False)
    assert {r.id for r in rules} == {a, b}
    notes = mem.search("note", k=5, filters=lambda r: r.meta.get("kind") == "note", touch=False)
    assert [r.id for r in notes] == [c]


def test_link_remove_and_errors():
    mem = Memory(clock=fixed_clock)
    r = mem.add("root", id="r")
    a = mem.add("child a", parent=r, id="a")
    b = mem.add("child b", parent=r, id="b")
    mem.link(b, a)
    assert mem.parent(b) == a and mem.level(b) == 2
    with pytest.raises(ValueError):
        mem.link(a, b)
    with pytest.raises(ValueError):
        mem.remove(a)
    mem.remove(a, cascade=True)
    assert len(mem) == 1 and mem.children(r) == []
    with pytest.raises(KeyError):
        mem.add("x", parent="missing")
    with pytest.raises(ValueError):
        mem.add("dup", id="r")
    with pytest.raises(ValueError):
        mem.search(mode="subtree")


def test_save_load_round_trip(tmp_path):
    mem = build_demo_memory(decay=HalfLife(hours=100), clock=fixed_clock)
    mem.search("hamstring")  # touch some records
    db = tmp_path / "m.db"
    mem.save(db)
    back = Memory.load(db, clock=fixed_clock)
    assert back.ids == mem.ids
    np.testing.assert_allclose(back.points, mem.points)
    assert back.get("hams").access_count == mem.get("hams").access_count
    assert isinstance(back.decay, HalfLife) and back.decay.hours == 100
    r1 = [r.id for r in mem.search("calf strain", k=3, touch=False)]
    r2 = [r.id for r in back.search("calf strain", k=3, touch=False)]
    assert r1 == r2


def test_inspection_outputs(tmp_path):
    mem = build_demo_memory(clock=fixed_clock)
    html = mem.inspect(tmp_path / "m.html", title="t")
    assert "<canvas" in html and '"nodes"' in html and (tmp_path / "m.html").exists()
    assert all(id_ in html for id_, _, _ in DEMO_TREE)
    rows = mem.table()
    assert len(rows) == len(mem) and {"id", "text", "depth", "weight"} <= set(rows[0])
    tree = mem.tree()
    assert "Human musculoskeletal system" in tree and "└──" in tree
    assert "<iframe" in mem._repr_html_()


def test_high_dim_disk_projection_keeps_depth():
    from poincare_memory.viz import disk_coordinates

    mem = build_demo_memory(clock=fixed_clock)
    xy = disk_coordinates(mem.points, mem.ball)
    r = np.linalg.norm(xy, axis=1)
    depth = mem.ball.distance_to_origin(mem.points)
    np.testing.assert_allclose(2 * np.arctanh(r), depth, atol=1e-9)


def test_structural_layout_keeps_depth_and_spreads_siblings():
    from poincare_memory.viz import structural_coordinates, to_dict

    mem = build_demo_memory(clock=fixed_clock)
    xy = structural_coordinates(mem)
    depth = mem.ball.distance_to_origin(mem.points)
    np.testing.assert_allclose(2 * np.arctanh(np.linalg.norm(xy, axis=1)), depth, atol=1e-6)
    kids = mem.children("lower")
    idx = {i: j for j, i in enumerate(mem.ids)}
    angles = np.sort([np.arctan2(xy[idx[k], 1], xy[idx[k], 0]) for k in kids])
    assert np.min(np.diff(angles)) > 0.15  # siblings are spread apart, not bunched
    assert to_dict(mem)["config"]["layout"] == "structural"
    assert to_dict(mem, layout="projection")["config"]["layout"] == "projection"


def test_add_many_matches_add_and_batches_embedding():
    calls = []

    class CountingEmbedder:
        dim = 64

        def embed(self, texts):
            calls.append(len(texts))
            from poincare_memory import HashingEmbedder

            return HashingEmbedder(dim=64).embed(texts)

    a = Memory(embedder=CountingEmbedder(), clock=fixed_clock)
    a.add("root", id="r")
    a.add("child one", parent="r", id="c1")
    a.add("child two", parent="r", id="c2")
    b = Memory(embedder=CountingEmbedder(), clock=fixed_clock)
    calls.clear()
    b.add_many(
        [
            {"text": "root", "id": "r"},
            {"text": "child one", "parent": "r", "id": "c1"},
            {"text": "child two", "parent": "r", "id": "c2"},
        ]
    )
    assert calls == [3]
    np.testing.assert_allclose(a.points, b.points)
    assert b.add_many([]) == []


def test_update_text_replaces_subtree_and_meta():
    mem = Memory(clock=fixed_clock)
    r = mem.add("root about knees", id="r", meta={"k": 1})
    c = mem.add("child", parent=r, id="c")
    before = mem.point(c).copy()
    mem.update(r, text="root about shoulders", meta={"k": 2}, valid_until=None)
    assert mem.get(r).meta == {"k": 2}
    assert not np.allclose(mem.point(c), before)  # child re-placed from the new parent direction
    assert mem.depth(c) == pytest.approx(mem.base_depth + mem.depth_step)
    with pytest.raises(ValueError):
        mem.add("   ")
    with pytest.raises(ValueError):
        mem.add("x", id="")


def test_json_round_trip_and_stats(tmp_path):
    mem = build_demo_memory(decay=HalfLife(hours=50), clock=fixed_clock)
    text = mem.to_json()
    back = Memory.from_json(text, clock=fixed_clock)
    assert back.ids == mem.ids
    np.testing.assert_allclose(back.points, mem.points)
    assert isinstance(back.decay, HalfLife) and back.decay.hours == 50
    mem.to_json(tmp_path / "m.json")
    again = Memory.from_json(tmp_path / "m.json", clock=fixed_clock)
    assert len(again) == len(mem)
    st = mem.stats()
    assert st["count"] == len(mem) and st["roots"] == 1 and st["max_level"] == 4
    assert "Memory(n=" in repr(mem)
    with pytest.raises(ValueError):
        Memory.from_json('{"config": {"format": 99}, "records": []}')


def test_index_search_matches_exact(tmp_path):
    mem = build_demo_memory(clock=fixed_clock)
    exact = [r.id for r in mem.search("calf strain soleus", k=5, touch=False)]
    mem.build_index(backend="numpy")
    indexed = [r.id for r in mem.search("calf strain soleus", k=5, touch=False)]
    assert exact == indexed
    forced = [
        r.id
        for r in mem.search("calf strain soleus", k=5, touch=False, use_index=True, oversample=20)
    ]
    assert forced == exact
    mem.add(
        "new note about soleus pain", parent="soleus"
    )  # marks the index dirty; rebuilt on demand
    assert mem._index_dirty
    rebuilt = mem.search("soleus pain", k=3, touch=False)
    assert rebuilt and not mem._index_dirty
    with pytest.raises(ValueError):
        mem.build_index(backend="nope")  # type: ignore[arg-type]


def test_memory_is_thread_safe():
    import threading

    mem = Memory(clock=fixed_clock)
    root = mem.add("root", id="r")

    def worker(n: int) -> None:
        for i in range(25):
            mem.add(f"worker {n} note {i}", parent=root)
            mem.search("note", k=3, touch=False)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(mem) == 101 and len(mem.children(root)) == 100
