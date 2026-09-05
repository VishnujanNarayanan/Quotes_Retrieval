import pytest
from retrieval import parse_advanced_query, strip_tag_phrases


@pytest.mark.parametrize("query,expected", [
    ("tagged with both 'love' and 'life'", ["love", "life"]),
    ('tagged with both "love" and "life"', ["love", "life"]),
    ("tagged with both love and life", ["love", "life"]),
    ("TAGGED WITH BOTH 'Love' AND 'Life'", ["love", "life"]),
])
def test_two_tag_filter(query, expected):
    assert parse_advanced_query(query) == expected


@pytest.mark.parametrize("query,expected", [
    ("tagged 'courage'", ["courage"]),
    ('tagged "courage"', ["courage"]),
    ("tagged courage", ["courage"]),
    ("Tagged Courage", ["courage"]),
])
def test_single_tag_filter(query, expected):
    assert parse_advanced_query(query) == expected


@pytest.mark.parametrize("query", [
    "",
    "courage in the face of fear",
    "wisdom by mark twain",
])
def test_no_filter_when_no_tag_phrase(query):
    assert parse_advanced_query(query) == []


def test_two_tag_form_wins_over_single():
    """'tagged with both X and Y' also matches the looser single-tag pattern.
    The two-tag branch must be tried first, or the filter silently narrows to X."""
    assert parse_advanced_query("tagged with both 'love' and 'life'") == ["love", "life"]


def test_tag_phrase_stripped_before_embedding():
    """The filter text must not reach the encoder, or it competes with the query."""
    assert strip_tag_phrases("courage tagged 'life'") == "courage"
    assert strip_tag_phrases("courage tagged with both 'love' and 'life'") == "courage"


def test_query_without_tag_phrase_is_untouched():
    assert strip_tag_phrases("courage in the face of fear") == "courage in the face of fear"


def test_tag_only_query_leaves_nothing_to_embed():
    """This is the case that routes search_quotes down its author-anchored branch."""
    assert strip_tag_phrases("tagged 'courage'") == ""
