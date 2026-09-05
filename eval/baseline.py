"""Score the fine-tuned encoder against a TF-IDF baseline.

    python eval/baseline.py                 # both systems
    python eval/baseline.py --tfidf-only    # skip the encoder (no torch needed)

The README lists "no retrieval evaluation" as a known limitation. This closes the
cheap half of it: a labelled query set, a lexical baseline, and recall@k for both.

Why a TF-IDF baseline and not a bigger model. Fine-tuning is only worth its cost
if it beats keyword matching, and nothing in the repo established that it does.
scikit-learn's TfidfVectorizer is the honest floor — it is what a keyword search
over the same corpus would give you, which is exactly the thing the project set
out to improve on.

The query set in queries.json is small and hand-labelled by reading the corpus.
It is a sanity check, not a benchmark: it catches a fine-tune that made retrieval
worse, which is the failure this repo could not previously see.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from retrieval import load_quotes_data  # noqa: E402

CORPUS = ROOT / "app" / "quotes.jsonl"
QUERIES = Path(__file__).parent / "queries.json"
MODEL_DIR = ROOT / "app" / "fine-tuned-quote-model"
KS = (1, 3, 5, 10)


def load_query_set():
    """[(query, {relevant author_keys})] — a hit is any quote by a listed author."""
    raw = json.loads(QUERIES.read_text(encoding="utf-8"))
    return [(q["query"], set(q["relevant_authors"])) for q in raw["queries"]]


def recall_at_k(ranked_authors, relevant, ks=KS):
    """Fraction of queries with at least one relevant author in the top k."""
    return {k: int(bool(set(ranked_authors[:k]) & relevant)) for k in ks}


def _report(name, per_query, ks=KS):
    print(f"\n{name}")
    print("  " + "  ".join(f"recall@{k}" for k in ks))
    line = []
    for k in ks:
        hits = sum(q[k] for q in per_query)
        line.append(f"{hits / len(per_query):>9.2f}")
    print("  " + "  ".join(line))


def score_tfidf(quotes, query_set):
    """Lexical floor: cosine similarity over word-level TF-IDF."""
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vec.fit_transform([q["quote"] for q in quotes])
    out = []
    for query, relevant in query_set:
        sims = cosine_similarity(vec.transform([query]), matrix).ravel()
        order = np.argsort(-sims)
        out.append(recall_at_k([quotes[i]["author_key"] for i in order], relevant))
    return out


def score_encoder(quotes, query_set):
    """The fine-tuned sentence-transformer, ranked by the same L2 the app uses."""
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(MODEL_DIR))
    emb = model.encode([q["quote"] for q in quotes], convert_to_tensor=True)
    emb = emb.cpu().detach().numpy().astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    out = []
    for query, relevant in query_set:
        qv = model.encode(query, convert_to_tensor=True)
        qv = qv.cpu().detach().numpy().astype("float32").reshape(1, -1)
        _, idx = index.search(qv, max(KS))
        out.append(recall_at_k([quotes[i]["author_key"] for i in idx[0]], relevant))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf-only", action="store_true",
                    help="skip the encoder, so the run needs no torch")
    args = ap.parse_args()

    quotes = load_quotes_data(CORPUS)
    query_set = load_query_set()
    print(f"corpus:  {len(quotes)} quotes")
    print(f"queries: {len(query_set)} labelled")

    _report("TF-IDF baseline (scikit-learn)", score_tfidf(quotes, query_set))

    if not args.tfidf_only:
        if not MODEL_DIR.exists():
            print(f"\nskipping the encoder: {MODEL_DIR} is not present")
            return 0
        _report("Fine-tuned sentence-transformer", score_encoder(quotes, query_set))
        print("\nThe encoder earns its cost only where it beats the baseline row above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
