"""Persisted corpus embeddings, with a fingerprint so a stale cache is never used.

Encoding all 2,508 quotes takes most of a cold start, and the result only changes
when the corpus text or the model changes. Caching it turns first paint from
"wait for the encoder" into "read a 3.7 MB array".

The fingerprint is the point. `quote_index.faiss`, the artefact the notebook left
behind, has the right shape — 2,508 vectors of 384 dimensions — but its vectors
disagree with the current corpus by up to 8.9e-4, because the notebook lowercased
its text and the corpus has since been mojibake-repaired. A cache that is loaded
whenever it *looks* right silently degrades every search. This one records what it
was built from and refuses itself when that no longer matches.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "quote_embeddings.npy"
META = HERE / "quote_embeddings.json"


def fingerprint(quotes_data, model_dir):
    """Identify the (corpus, model) pair these vectors were produced from.

    Covers the corpus text itself and the model weights, which are the only two
    inputs to the encoding. A change in either invalidates the cache.
    """
    h = hashlib.sha256()
    for q in quotes_data:
        h.update(q["quote"].encode("utf-8"))
        h.update(b"\0")
    corpus_digest = h.hexdigest()

    weights = Path(model_dir) / "model.safetensors"
    m = hashlib.sha256()
    if weights.exists():
        with open(weights, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                m.update(chunk)

    return {
        "corpus_sha256": corpus_digest,
        "corpus_records": len(quotes_data),
        "model_sha256": m.hexdigest(),
    }


def load(quotes_data, model_dir):
    """The cached embedding matrix, or None if there is no valid cache.

    Returns None rather than raising on every failure mode — a missing, corrupt,
    outdated or shape-mismatched cache all mean the same thing to the caller:
    encode the corpus instead.
    """
    if not (CACHE.exists() and META.exists()):
        return None
    try:
        stored = json.loads(META.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != fingerprint(quotes_data, model_dir):
            return None
        emb = np.load(CACHE)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    if emb.shape[0] != len(quotes_data):
        return None
    return np.ascontiguousarray(emb, dtype="float32")


def save(embeddings, quotes_data, model_dir):
    """Write the cache and the fingerprint that validates it.

    Best effort: a read-only or full filesystem costs a slow start, not a failure,
    so a write error is swallowed rather than taking the app down with it.
    """
    try:
        np.save(CACHE, np.ascontiguousarray(embeddings, dtype="float32"))
        META.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint(quotes_data, model_dir),
                    "shape": list(embeddings.shape),
                    "note": "Regenerate with: python app/build_embeddings.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True
