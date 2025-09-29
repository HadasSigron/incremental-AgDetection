import json
import time
from types import SimpleNamespace
from pathlib import Path

from core.eval.incremental import evaluate_incremental

class DummyDataset:
    """Yields a small fixed set of samples."""
    def __init__(self, root: Path, n=3):
        self.samples = []
        (root / "data").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            p = root / "data" / f"img_{i+1}.jpg"
            if not p.exists():
                p.write_bytes(b"X")  # tiny file so hashing has something to read
            self.samples.append({"id": i + 1, "path": str(p), "label": i % 2})

    def iter_samples(self, split="val"):
        for s in self.samples:
            yield s

class CountingRunner:
    """Runner that returns a trivial prediction and counts calls."""
    def __init__(self):
        self.calls = 0

    def infer_one(self, sample):
        self.calls += 1
        # trivial "prediction" structure
        return {"id": sample["id"], "pred_label": sample.get("label", 0), "score": 0.5}

def _mk_cfg(tmp_path: Path):
    # minimal cfg shape expected by evaluate_incremental
    weights = tmp_path / "models" / "w.onnx"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        weights.write_bytes(b"W")
    return SimpleNamespace(
        task=SimpleNamespace(value="classification"),
        domain=SimpleNamespace(value="fruit"),
        benchmark_name="DUMMY",
        paths=SimpleNamespace(runs_root=tmp_path),
        model=SimpleNamespace(weights=weights),
        eval=SimpleNamespace(metrics=["accuracy"]),  # placeholder
    )

def test_incremental_creates_artifacts_and_uses_cache(tmp_path):
    cfg = _mk_cfg(tmp_path)
    ds = DummyDataset(tmp_path, n=4)
    runner = CountingRunner()

    # First run: should call runner for all samples, create artifacts
    metrics1 = evaluate_incremental(cfg, ds, runner, cancel=None)
    assert runner.calls == 4, "first run must infer all samples"
    # artifacts exist
    # find most recent run dir
    runs_root = tmp_path / "classification" / "fruit" / "DUMMY"
    run_dirs = sorted(runs_root.iterdir())
    assert run_dirs, "run directory should be created"
    latest = run_dirs[-1]
    assert (latest / "predictions.json").exists()
    assert (latest / "metrics.json").exists()
    assert (latest / "config.used.json").exists()
    # check metrics shape
    assert "num_predictions" in metrics1 or "status" in metrics1

    # Second run: must hit the cache (0 new calls) if nothing changed
    runner2 = CountingRunner()
    metrics2 = evaluate_incremental(cfg, ds, runner2, cancel=None)
    assert runner2.calls == 0, "second run should reuse cached predictions"

    # Change one image so only one sample is recomputed
    # Touch the file to change mtime/size
    img_to_change = Path(ds.samples[0]["path"])
    time.sleep(0.02)  # ensure mtime difference on fast filesystems
    img_to_change.write_bytes(b"XX")  # change size → different file_sig
    runner3 = CountingRunner()
    metrics3 = evaluate_incremental(cfg, ds, runner3, cancel=None)
    assert runner3.calls == 1, "only the changed sample should be recomputed"
