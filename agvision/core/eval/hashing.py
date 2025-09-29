"""Per-sample and config hashing utilities for incremental evaluation."""
from pathlib import Path
import hashlib
import json

def file_sig(p: Path) -> str:
    """
    Create a quick signature for a file:
    absolute path + last modified time (mtime) + file size (bytes).
    If the file changes in any way, this signature changes.
    """
    p = Path(p).expanduser().resolve()
    st = p.stat()
    return f"{p.as_posix()}|{int(st.st_mtime)}|{st.st_size}"

def dict_sig(d: dict) -> str:
    """
    Create a stable hash for a Python dict (e.g., config):
    dump to JSON with sorted keys, then compute SHA1.
    Any value change → different hash.
    """
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def model_sig(weights: str | Path | None) -> str:
    """
    Create a signature for model weights file.
    If no weights were provided, return a constant 'noweights'.
    Otherwise reuse file_sig(weights).
    """
    if not weights:
        return "noweights"
    return file_sig(Path(weights))

def sample_key(img_path: str | Path, model_signature: str, config_signature: str) -> str:
    """
    Build the unique key for one dataset sample by combining:
    - file signature of the image
    - model signature
    - config signature
    Then hash the combination into a final SHA1 string.
    If any part changes, the key changes.
    """
    raw = f"{file_sig(Path(img_path))}|{model_signature}|{config_signature}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
