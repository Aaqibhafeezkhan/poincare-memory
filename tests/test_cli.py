from poincare_memory.cli import main


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
