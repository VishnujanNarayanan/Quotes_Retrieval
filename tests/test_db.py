import db


def test_queries_file_parses_into_named_statements():
    q = db.load_queries()
    assert {"author_catalogue", "ordinals_with_all_tags", "quotes_by_author",
            "tag_frequencies", "corpus_summary"} <= set(q)
    assert all(v.strip() for v in q.values())


def test_author_catalogue_counts_and_orders(con):
    cat = db.author_catalogue(con)
    assert cat[0] == ("Shakespeare", 2)
    assert ("Cicero", 1) in cat


def test_author_catalogue_uses_the_stripped_key(con):
    """'Hemingway,' in the corpus must appear as 'Hemingway' or the picker's
    value never matches a record."""
    assert ("Hemingway", 1) in db.author_catalogue(con)


def test_single_tag_filter(con):
    assert db.ordinals_with_all_tags(con, ["books"]) == [4]


def test_two_tag_filter_is_an_intersection_not_a_union(con):
    """'love' matches records 2 and 3, 'life' matches 0, 2 and 5; only record 2
    carries both. A union here would quietly widen every two-tag search."""
    assert db.ordinals_with_all_tags(con, ["love", "life"]) == [2]


def test_tag_filter_is_case_insensitive(con):
    assert db.ordinals_with_all_tags(con, ["LOVE", "Life"]) == [2]


def test_repeated_tag_does_not_break_the_count(con):
    assert db.ordinals_with_all_tags(con, ["love", "love"]) == [2, 3]


def test_unknown_tag_matches_nothing(con):
    assert db.ordinals_with_all_tags(con, ["sailing"]) == []


def test_no_tags_means_no_filter(con):
    assert db.ordinals_with_all_tags(con, []) is None


def test_quotes_by_author(con):
    assert db.quotes_by_author(con, "Shakespeare") == [2, 3]
    assert db.quotes_by_author(con, "Nobody") == []


def test_ordinals_are_corpus_positions(con, corpus):
    """The join between SQL and FAISS: ordinal N must be record N."""
    rows = list(con.execute("SELECT ordinal, quote FROM quotes ORDER BY ordinal"))
    assert [r["quote"] for r in rows] == [q["quote"] for q in corpus]


def test_tag_frequencies(con):
    freqs = dict(db.tag_frequencies(con))
    assert freqs["life"] == 3
    assert freqs["love"] == 2


def test_corpus_summary(con, corpus):
    s = db.corpus_summary(con)
    assert s["quotes"] == len(corpus)
    assert s["authors"] == 5
    assert s["tags"] == 5


def test_build_is_idempotent(tmp_path, corpus):
    """The schema drops before it creates, so a rebuild over a live file must
    not double the rows."""
    p = tmp_path / "twice.db"
    db.build(corpus, db_path=p).close()
    con = db.build(corpus, db_path=p)
    assert db.corpus_summary(con)["quotes"] == len(corpus)
    con.close()
