from core.eval.store import CacheStore

def test_cache_store_put_get(tmp_path):
    db = tmp_path / ".cache.sqlite"
    store = CacheStore(db)
    key = "abc123"
    val = {"id": 1, "pred": 0.9}
    assert store.get(key) is None, "empty store should return None"
    store.put(key, val)
    got = store.get(key)
    assert got == val, "stored value must be returned intact"

def test_cache_store_overwrite(tmp_path):
    db = tmp_path / ".cache.sqlite"
    store = CacheStore(db)
    key = "dup"
    store.put(key, {"v": 1})
    store.put(key, {"v": 2})
    got = store.get(key)
    assert got == {"v": 2}, "REPLACE must overwrite the previous value"
