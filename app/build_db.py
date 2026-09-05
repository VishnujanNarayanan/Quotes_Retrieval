"""Build quotes.db from quotes.jsonl.

    python build_db.py

The app builds the database itself on first run if it is missing, so this script
is for rebuilding it deliberately — after clean_quotes.py has repaired the corpus,
or to inspect the SQL by hand:

    sqlite3 quotes.db \
        "SELECT author_key, COUNT(*) n FROM quotes GROUP BY 1 ORDER BY n DESC LIMIT 10;"
"""

import sys
from pathlib import Path

import db
from retrieval import load_quotes_data

CORPUS = Path(__file__).parent / "quotes.jsonl"


def main():
    quotes_data = load_quotes_data(CORPUS)
    con = db.build(quotes_data)
    summary = db.corpus_summary(con)
    print(f"wrote {db.DB_PATH.name}")
    print(f"  quotes:  {summary['quotes']}")
    print(f"  authors: {summary['authors']}")
    print(f"  tags:    {summary['tags']}")

    top = db.author_catalogue(con)[:5]
    print("  most quoted:")
    for name, n in top:
        print(f"    {n:>4}  {name}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
