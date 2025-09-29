from types import SimpleNamespace
from pathlib import Path
from core.eval.runs import make_run_dir

def test_make_run_dir_structure(tmp_path, monkeypatch):
    # Build a minimal cfg with required attributes
    cfg = SimpleNamespace(
        task=SimpleNamespace(value="classification"),
        domain=SimpleNamespace(value="fruit"),
        benchmark_name="DUMMY",
        paths=SimpleNamespace(runs_root=tmp_path),
    )
    run_dir = make_run_dir(cfg)
    # Expected layout: <root>/classification/fruit/DUMMY/<timestamp>
    parts = run_dir.relative_to(tmp_path).parts
    assert parts[0] == "classification"
    assert parts[1] == "fruit"
    assert parts[2] == "DUMMY"
    assert len(parts[3]) >= 8, "timestamp folder must exist"
    assert run_dir.exists() and run_dir.is_dir()
