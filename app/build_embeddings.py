"""Encode the corpus once and write the cache the app loads at startup.

    python app/build_embeddings.py

Run this after changing the corpus or retraining the model. The app falls back to
encoding in-process when the cache is missing or its fingerprint no longer matches,
so forgetting to run it costs a slow first load rather than wrong results.
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import embed_cache  # noqa: E402
from retrieval import load_quotes_data  # noqa: E402

CORPUS = HERE / "quotes.jsonl"
MODEL_DIR = HERE / "fine-tuned-quote-model"


def main():
    from sentence_transformers import SentenceTransformer

    quotes = load_quotes_data(CORPUS)
    print(f"corpus: {len(quotes)} quotes")

    model = SentenceTransformer(str(MODEL_DIR))
    start = time.perf_counter()
    emb = model.encode([q["quote"] for q in quotes], convert_to_tensor=True)
    emb = emb.cpu().detach().numpy().astype("float32")
    print(f"encoded in {time.perf_counter() - start:.1f}s -> {emb.shape}")

    if embed_cache.save(emb, quotes, MODEL_DIR):
        size = embed_cache.CACHE.stat().st_size / 1e6
        print(f"wrote {embed_cache.CACHE.name} ({size:.1f} MB) and {embed_cache.META.name}")
    else:
        print("could not write the cache", file=sys.stderr)
        return 1

    # Prove the cache is loadable and validates, rather than assuming it.
    check = embed_cache.load(quotes, MODEL_DIR)
    assert check is not None, "cache did not validate immediately after writing"
    assert check.shape == emb.shape, "cache shape does not match what was encoded"
    print("verified: cache reloads and its fingerprint matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
