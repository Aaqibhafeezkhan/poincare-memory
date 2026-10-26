from poincare_memory.cli import main
from poincare_memory.store.memory import Memory


def test_cli_demo_inspect_tree_search(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "-o", str(out)]) == 0
    assert (tmp_path / "demo.db").exists() and (tmp_path / "demo.html").exists()
    assert main(["tree", str(tmp_path / "demo.db")]) == 0
    assert main(["search", str(tmp_path / "demo.db"), "calf strain", "-k", "2"]) == 0
    assert main(["inspect", str(tmp_path / "demo.db"), "-o", str(tmp_path / "x.html")]) == 0
    assert main(["stats", str(tmp_path / "demo.db")]) == 0
    assert main(["export", str(tmp_path / "demo.db"), str(tmp_path / "demo.json")]) == 0
    assert main(["import", str(tmp_path / "demo.json"), str(tmp_path / "demo2.db")]) == 0
    captured = capsys.readouterr().out
    assert "Soleus" in captured and "wrote" in captured and '"count"' in captured


def test_cli_tree_root_prints_only_selected_subtree(tmp_path, capsys):
    mem = Memory()
    mem.add("Body", id="body")
    mem.add("Hamstring group", parent="body", id="hams")
    mem.add("Biceps femoris", parent="hams", id="biceps")
    mem.add("Calf muscles", parent="body", id="calves")
    db = tmp_path / "memory.db"
    mem.save(db)

    assert main(["tree", str(db), "--root", "hams"]) == 0

    output = capsys.readouterr().out
    assert "Hamstring group" in output
    assert "Biceps femoris" in output
    assert "Body" not in output
    assert "Calf muscles" not in output
