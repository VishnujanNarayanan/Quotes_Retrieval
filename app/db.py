"""SQLite access for the quote corpus.

Every statement lives in queries.sql, not in a Python string, so the SQL can be
read and run on its own. This module only loads them, binds parameters and
returns rows.
"""

import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "quotes.db"
SCHEMA_PATH = HERE / "schema.sql"
QUERIES_PATH = HERE / "queries.sql"

_NAME = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def load_queries(path=QUERIES_PATH):
    """Parse queries.sql into {name: sql}."""
    text = Path(path).read_text(encoding="utf-8")
    marks = list(_NAME.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.end():end].strip()
    return out


QUERIES = load_queries()


def connect(db_path=DB_PATH, check_same_thread=True):
    """Read-mostly connection. Streamlit caches one connection and reruns the
    script on worker threads, so the app passes check_same_thread=False."""
    con = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    return con


def build(quotes_data, db_path=DB_PATH, schema_path=SCHEMA_PATH):
    """(Re)create the database from the records load_quotes_data() returned.

    `ordinal` is the record's position in that list, which is also its row in the
    embedding matrix — that is the join between SQL filtering and vector search.
    """
    con = connect(db_path)
    with con:
        con.executescript(Path(schema_path).read_text(encoding="utf-8"))
        con.executemany(
            "INSERT INTO quotes (ordinal, quote, author, author_key) VALUES (?, ?, ?, ?)",
            [(i, q["quote"], q["author"], q["author_key"])
             for i, q in enumerate(quotes_data)],
        )
        con.executemany(
            "INSERT OR IGNORE INTO quote_tags (ordinal, tag) VALUES (?, ?)",
            [(i, t) for i, q in enumerate(quotes_data) for t in q["tags_lc"]],
        )
    return con


def author_catalogue(con):
    """[(author_key, count)] ordered by count desc, for the picker."""
    return [(r["author_key"], r["n"])
            for r in con.execute(QUERIES["author_catalogue"])]


def ordinals_with_all_tags(con, tags):
    """Corpus positions carrying EVERY tag in `tags` (the intersection)."""
    if not tags:
        return None
    tags = [t.lower() for t in tags]
    sql = QUERIES["ordinals_with_all_tags"] % ",".join("?" * len(tags))
    return [r["ordinal"] for r in con.execute(sql, (*tags, len(set(tags))))]


def quotes_by_author(con, author_key):
    return [r["ordinal"] for r in con.execute(QUERIES["quotes_by_author"], (author_key,))]


def tag_frequencies(con):
    return [(r["tag"], r["n"]) for r in con.execute(QUERIES["tag_frequencies"])]


def corpus_summary(con):
    r = con.execute(QUERIES["corpus_summary"]).fetchone()
    return {"quotes": r["quotes"], "authors": r["authors"], "tags": r["tags"]}
