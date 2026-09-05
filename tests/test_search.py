
from retrieval import search_quotes


def _ids(results, corpus):
    return [next(i for i, q in enumerate(corpus) if q["quote"] == r["quote"])
            for r in results]


def test_ranks_by_distance(corpus, index, vectors, embed):
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=3)
    assert _ids(got, corpus) == [0, 1, 2]


def test_top_k_is_respected(corpus, index, vectors, embed):
    assert len(search_quotes("courage", corpus, index, vectors, embed, top_k=2)) == 2
    assert len(search_quotes("courage", corpus, index, vectors, embed, top_k=6)) == 6


def test_top_k_larger_than_corpus_returns_everything(corpus, index, vectors, embed):
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=99)
    assert len(got) == len(corpus)


def test_no_author_means_no_fillers(corpus, index, vectors, embed):
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=4)
    assert not any(r["is_filler"] for r in got)


def test_trailing_comma_stripped_from_author(corpus, index, vectors, embed):
    """The scrape leaves 'Hemingway,'; the card and the picker must agree on
    'Hemingway' or anchoring silently matches nothing."""
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=1)
    assert got[0]["author"] == "Hemingway"


def test_author_anchor_puts_that_author_first(corpus, index, vectors, embed):
    """Hemingway holds the closest vector, so his quote clears the percentile
    cutoff and must lead as a primary rather than arrive as a filler."""
    got = search_quotes("courage", corpus, index, vectors, embed,
                        author="Hemingway", top_k=4)
    primary = [r for r in got if not r["is_filler"]]
    assert primary, "an on-topic Hemingway quote should have been primary"
    assert all(r["author"] == "Hemingway" for r in primary)
    assert got[0] is primary[0], "primaries render before fillers"


def test_remaining_slots_are_filled_and_flagged(corpus, index, vectors, embed):
    """An author with too few on-topic quotes gets backfilled from others, and
    those must be visibly secondary rather than passed off as the author's."""
    got = search_quotes("love", corpus, index, vectors, embed,
                        author="Cicero", top_k=4)
    assert len(got) == 4
    fillers = [r for r in got if r["is_filler"]]
    assert fillers
    assert all(r["author"] != "Cicero" for r in fillers)


def test_fillers_never_duplicate_the_anchored_author(corpus, index, vectors, embed):
    got = search_quotes("love", corpus, index, vectors, embed,
                        author="Shakespeare", top_k=6)
    assert all(not r["is_filler"] or r["author"] != "Shakespeare" for r in got)


def test_percentile_cutoff_excludes_far_author_quotes(corpus, index, vectors, embed):
    """Lincoln's only quote is the furthest vector in the fixture, so it must not
    come back as a primary match — that is the whole point of the cutoff."""
    got = search_quotes("anything", corpus, index, vectors, embed,
                        author="Lincoln", top_k=3)
    primary = [r for r in got if not r["is_filler"]]
    assert primary == []


def test_author_only_query_reports_no_distance(corpus, index, vectors, embed):
    """With no query there is no distance to show, and a meter drawn from a
    fabricated one would mislead."""
    got = search_quotes("", corpus, index, vectors, embed,
                        author="Shakespeare", top_k=2)
    assert got and all(r["distance"] is None for r in got)
    assert all(r["author"] == "Shakespeare" for r in got)


def test_author_only_backfills_from_the_centroid(corpus, index, vectors, embed):
    got = search_quotes("", corpus, index, vectors, embed, author="Cicero", top_k=3)
    assert len(got) == 3
    assert got[0]["author"] == "Cicero" and not got[0]["is_filler"]
    assert all(r["is_filler"] for r in got[1:])


def test_empty_query_and_no_author_returns_nothing(corpus, index, vectors, embed):
    assert search_quotes("", corpus, index, vectors, embed) == []


def test_unknown_author_returns_only_fillers(corpus, index, vectors, embed):
    got = search_quotes("courage", corpus, index, vectors, embed,
                        author="Nobody", top_k=3)
    assert len(got) == 3
    assert all(r["is_filler"] for r in got)


def test_python_tag_filter_restricts_the_pool(corpus, index, vectors, embed):
    got = search_quotes("love tagged 'books'", corpus, index, vectors, embed, top_k=5)
    assert [r["author"] for r in got] == ["Cicero"]


def test_tag_filter_matching_nothing_returns_empty(corpus, index, vectors, embed):
    assert search_quotes("tagged 'sailing'", corpus, index, vectors, embed) == []


def test_eligible_ids_override_the_python_filter(corpus, index, vectors, embed):
    """This is the seam the SQL tag filter plugs into."""
    got = search_quotes("courage", corpus, index, vectors, embed,
                        top_k=5, eligible_ids=[2, 3])
    assert sorted(_ids(got, corpus)) == [2, 3]


def test_distances_are_plain_floats(corpus, index, vectors, embed):
    """np.float32 leaks into the f-string in the card and renders badly."""
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=2)
    assert all(type(r["distance"]) is float for r in got)


def test_results_carry_the_original_casing(corpus, index, vectors, embed):
    got = search_quotes("courage", corpus, index, vectors, embed, top_k=1)
    assert got[0]["quote"] == "Courage is grace under pressure."
    assert got[0]["tags"] == ["courage", "life"]
