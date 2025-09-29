import time
from pathlib import Path
from core.eval.hashing import file_sig, dict_sig, model_sig, sample_key

def test_file_sig_changes_on_modify(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    sig1 = file_sig(p)
    time.sleep(0.01)  # ensure mtime differs on fast filesystems
    p.write_text("hello world")
    sig2 = file_sig(p)
    assert sig1 != sig2, "file_sig must change when file content changes"

def test_dict_sig_changes_on_value():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 3}
    assert dict_sig(d1) != dict_sig(d2), "dict_sig must change when dict values change"

def test_model_sig_when_missing_and_present(tmp_path):
    assert model_sig(None) == "noweights"
    w = tmp_path / "w.bin"
    w.write_bytes(b"X")
    s1 = model_sig(w)
    w.write_bytes(b"XY")
    s2 = model_sig(w)
    assert s1 != s2, "model_sig must change when weights file changes"

def test_sample_key_combines_all(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"IMG")
    cfg1 = dict_sig({"x": 1})
    cfg2 = dict_sig({"x": 2})
    m = "MODEL"
    k1 = sample_key(img, m, cfg1)
    k2 = sample_key(img, m, cfg2)
    assert k1 != k2, "sample_key must change when config signature changes"
