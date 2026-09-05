import json

import embed_cache
import numpy as np
import pytest


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the cache at a temp dir so tests never touch the committed one."""
    monkeypatch.setattr(embed_cache, "CACHE", tmp_path / "emb.npy")
    monkeypatch.setattr(embed_cache, "META", tmp_path / "emb.json")
    return tmp_path


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"pretend weights")
    return d


@pytest.fixture
def vecs(corpus):
    return np.arange(len(corpus) * 4, dtype="float32").reshape(len(corpus), 4)


def test_round_trips(cache_dir, model_dir, corpus, vecs):
    assert embed_cache.save(vecs, corpus, model_dir)
    np.testing.assert_array_equal(embed_cache.load(corpus, model_dir), vecs)


def test_missing_cache_returns_none(cache_dir, model_dir, corpus):
    assert embed_cache.load(corpus, model_dir) is None


def test_changed_corpus_text_invalidates(cache_dir, model_dir, corpus, vecs):
    """The failure this whole module exists to prevent: the committed
    quote_index.faiss had the right shape but vectors from a different corpus."""
    embed_cache.save(vecs, corpus, model_dir)
    changed = [dict(q) for q in corpus]
    changed[0]["quote"] = "something else entirely"
    assert embed_cache.load(changed, model_dir) is None


def test_dropped_record_invalidates(cache_dir, model_dir, corpus, vecs):
    embed_cache.save(vecs, corpus, model_dir)
    assert embed_cache.load(corpus[:-1], model_dir) is None


def test_retrained_model_invalidates(cache_dir, model_dir, corpus, vecs, tmp_path):
    embed_cache.save(vecs, corpus, model_dir)
    (model_dir / "model.safetensors").write_bytes(b"different weights")
    assert embed_cache.load(corpus, model_dir) is None


def test_reordered_corpus_invalidates(cache_dir, model_dir, corpus, vecs):
    """Row N of the matrix must be record N; reordering breaks that silently."""
    embed_cache.save(vecs, corpus, model_dir)
    assert embed_cache.load(list(reversed(corpus)), model_dir) is None


def test_corrupt_metadata_returns_none(cache_dir, model_dir, corpus, vecs):
    embed_cache.save(vecs, corpus, model_dir)
    embed_cache.META.write_text("{not json", encoding="utf-8")
    assert embed_cache.load(corpus, model_dir) is None


def test_corrupt_array_returns_none(cache_dir, model_dir, corpus, vecs):
    embed_cache.save(vecs, corpus, model_dir)
    embed_cache.CACHE.write_bytes(b"not a numpy file")
    assert embed_cache.load(corpus, model_dir) is None


def test_shape_mismatch_returns_none(cache_dir, model_dir, corpus, vecs):
    """Fingerprint intact but the array is the wrong height — belt and braces,
    because a row/record mismatch would index the wrong quote."""
    embed_cache.save(vecs, corpus, model_dir)
    np.save(embed_cache.CACHE, vecs[:-1])
    assert embed_cache.load(corpus, model_dir) is None


def test_load_returns_float32_contiguous(cache_dir, model_dir, corpus, vecs):
    """FAISS requires both; a non-contiguous float64 array raises at add()."""
    embed_cache.save(vecs.astype("float64"), corpus, model_dir)
    out = embed_cache.load(corpus, model_dir)
    assert out.dtype == np.dtype("float32")
    assert out.flags["C_CONTIGUOUS"]


def test_metadata_records_the_shape(cache_dir, model_dir, corpus, vecs):
    embed_cache.save(vecs, corpus, model_dir)
    meta = json.loads(embed_cache.META.read_text(encoding="utf-8"))
    assert meta["shape"] == list(vecs.shape)
    assert set(meta["fingerprint"]) == {"corpus_sha256", "corpus_records", "model_sha256"}


def test_fingerprint_is_stable_across_calls(cache_dir, model_dir, corpus):
    assert embed_cache.fingerprint(corpus, model_dir) == embed_cache.fingerprint(corpus, model_dir)


def test_save_survives_an_unwritable_location(model_dir, corpus, vecs, monkeypatch, tmp_path):
    """A read-only filesystem should cost a slow start, not crash the app."""
    monkeypatch.setattr(embed_cache, "CACHE", tmp_path / "nope" / "emb.npy")
    monkeypatch.setattr(embed_cache, "META", tmp_path / "nope" / "emb.json")
    assert embed_cache.save(vecs, corpus, model_dir) is False
