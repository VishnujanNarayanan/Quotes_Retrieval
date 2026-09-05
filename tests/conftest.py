import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import db as db_module  # noqa: E402
import retrieval  # noqa: E402


def _record(quote, author, tags):
    return {
        "quote": quote,
        "author": author,
        "tags": tags,
        "quote_lc": quote.lower(),
        "author_lc": author.lower(),
        "author_key": author.rstrip(",").strip(),
        "tags_lc": [t.lower() for t in tags],
    }


@pytest.fixture
def corpus():
    """Six records with hand-placed vectors, so every ranking assertion below is
    a statement about the retrieval rules rather than about the model."""
    return [
        _record("Courage is grace under pressure.", "Hemingway,", ["courage", "life"]),
        _record("Fear is the mind-killer.", "Herbert", ["courage", "fear"]),
        _record("Love all, trust a few.", "Shakespeare", ["love", "life"]),
        _record("The course of true love never did run smooth.", "Shakespeare", ["love"]),
        _record("A room without books is a body without a soul.", "Cicero", ["books"]),
        _record("Whatever you are, be a good one.", "Lincoln", ["life"]),
    ]


@pytest.fixture
def vectors():
    """2-D embeddings: index 0 sits at the origin, and each later record is
    further from it than the last, so nearest-neighbour order is 0,1,2,3,4,5."""
    return np.array(
        [[0.0, 0.0],
         [0.1, 0.0],
         [0.3, 0.0],
         [0.6, 0.0],
         [1.0, 0.0],
         [1.5, 0.0]],
        dtype="float32",
    )


@pytest.fixture
def index(vectors):
    return retrieval.build_index(vectors)


@pytest.fixture
def embed():
    """Every query embeds to the origin, so distance is purely the record's own
    position — the fixture chooses the ranking, not a model."""
    return lambda text: np.array([[0.0, 0.0]], dtype="float32")


@pytest.fixture
def con(tmp_path, corpus):
    """A real SQLite database over the fixture corpus."""
    c = db_module.build(corpus, db_path=tmp_path / "test.db")
    yield c
    c.close()
