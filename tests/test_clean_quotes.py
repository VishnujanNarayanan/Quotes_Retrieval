import json

import clean_quotes
import pytest


@pytest.mark.parametrize("broken,intended", [
    ("Itâ€™s a truth", "It’s a truth"),
    ("â€œquotedâ€\x9d", "“quoted”"),
    ("an emâ€”dash", "an em—dash"),
])
def test_reverses_cp1252_mojibake(broken, intended):
    assert clean_quotes.repair(broken) == intended


def test_leaves_clean_text_alone():
    """Anything that survives untouched must come back byte-identical, or a
    'repair' run silently rewrites correct records."""
    for text in ["It's plain ASCII.", "Curly ’quotes’ are already correct.",
                 "Accents like café are fine.", ""]:
        assert clean_quotes.repair(text) == text


def test_curly_quotation_marks_are_not_straightened():
    """fix_encoding, not fix_text: the corpus's typography is deliberate."""
    assert clean_quotes.repair("“already right”") == "“already right”"


def test_destroyed_bytes_are_substituted():
    """windows-1252 left these undefined, so the original byte is gone and ftfy
    cannot recover it; the mapping is the only way back."""
    for broken, intended in clean_quotes.DESTROYED_BYTES.items():
        assert intended in clean_quotes.repair(f"x{broken}y")


def test_repairs_damage_embedded_in_otherwise_valid_text():
    """ftfy declines a string that is only partly damaged, which is exactly the
    shape of the Arabic records — repair_fragments is what covers them."""
    out = clean_quotes.repair_fragments("Ø§Ù„Ø­Ø¨ and English")
    assert "and English" in out
    assert "Ø" not in out


def test_clean_record_reports_which_fields_changed():
    rec = {"quote": "Itâ€™s here", "author": "Plainâ€™s", "tags": ["loveâ€™s"]}
    out, changed = clean_quotes.clean_record(rec)
    assert set(changed) == {"quote", "author", "tags"}
    assert out["quote"] == "It’s here"
    assert out["tags"] == ["love’s"]


def test_clean_record_reports_nothing_for_clean_input():
    rec = {"quote": "Fine.", "author": "Nobody", "tags": ["life"]}
    out, changed = clean_quotes.clean_record(rec)
    assert changed == []
    assert out == rec


def test_clean_record_preserves_the_field_set():
    """main() asserts on this; a repair that adds or drops a key must fail loudly
    rather than write a corpus the loader cannot read."""
    rec = {"quote": "Itâ€™s here", "author": "X", "tags": [], "extra": 1}
    out, _ = clean_quotes.clean_record(rec)
    assert set(out.keys()) == set(rec.keys())


def test_clean_record_never_changes_tag_count():
    rec = {"quote": "x", "author": "y", "tags": ["aâ€™", "b", "c"]}
    out, _ = clean_quotes.clean_record(rec)
    assert len(out["tags"]) == 3


def test_load_reads_jsonl_and_skips_blank_lines(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"quote": "a", "author": "b", "tags": []}\n\n'
                 '{"quote": "c", "author": "d", "tags": ["e"]}\n', encoding="utf-8")
    recs = clean_quotes.load(p)
    assert len(recs) == 2
    assert recs[1]["tags"] == ["e"]


def test_round_trips_through_json(tmp_path):
    """The writer uses ensure_ascii=False; a repaired record must survive the
    write/re-read cycle main() verifies."""
    rec = {"quote": clean_quotes.repair("Itâ€™s here"), "author": "X", "tags": ["café"]}
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert clean_quotes.load(p) == [rec]
