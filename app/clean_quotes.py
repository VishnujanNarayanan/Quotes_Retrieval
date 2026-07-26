"""Repair mojibake in quotes.jsonl.

The corpus was scraped with a UTF-8 payload decoded as cp1252, so characters like
' became a€™ and - became a€". ftfy reverses that transformation.

The script is deliberately conservative: it only rewrites fields whose text ftfy
changes, it never drops a record, and it refuses to write if the record count or
any quote's emptiness changes.

    python clean_quotes.py            # report what would change
    python clean_quotes.py --apply    # back up, then rewrite
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import ftfy
import ftfy.bad_codecs  # registers the sloppy-windows-1252 codec  # noqa: F401

DATA = Path(__file__).parent / "quotes.jsonl"
BACKUP = DATA.with_suffix(".jsonl.bak")


# Bytes that windows-1252 leaves undefined were replaced by U+FFFD during the
# original bad decode, so the information is gone and ftfy cannot reverse it.
# Only two such sequences occur in this corpus, and context makes both certain:
#   "a<U+20AC><U+FFFD>" closes a quotation that opened with a left double quote -> byte 0x9D -> U+201D
#   "U<U+FFFD>" sits inside Arabic words that only parse with feh          -> byte 0x81 -> U+0641
DESTROYED_BYTES = {
    "â€�": "”",
    "Ù�": "ف",
}


def _sloppy_encodable(ch):
    try:
        ch.encode("sloppy-windows-1252")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def repair_fragments(text):
    """Fix mojibake runs embedded inside otherwise-valid text.

    ftfy declines to touch a string when only part of it is damaged, which is the
    case for the Arabic entries. This walks each run of non-ASCII characters that
    windows-1252 could have produced and re-decodes it as UTF-8, keeping the run
    only when it decodes cleanly.
    """
    out = []
    i = 0
    while i < len(text):
        if ord(text[i]) > 127 and _sloppy_encodable(text[i]):
            j = i
            while j < len(text) and ord(text[j]) > 127 and _sloppy_encodable(text[j]):
                j += 1
            run = text[i:j]
            try:
                decoded = run.encode("sloppy-windows-1252").decode("utf-8")
                out.append(decoded)
            except (UnicodeEncodeError, UnicodeDecodeError):
                out.append(run)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def repair(text):
    """Undo encoding damage only.

    fix_encoding is used rather than fix_text because fix_text also straightens
    curly quotation marks, and the corpus's typographic quotes are correct as-is.
    """
    text = ftfy.fix_encoding(text)
    for broken, intended in DESTROYED_BYTES.items():
        text = text.replace(broken, intended)
    return repair_fragments(text)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clean_record(rec):
    """Return (cleaned_record, list_of_field_names_changed)."""
    out = dict(rec)
    changed = []

    for field in ("quote", "author"):
        original = rec.get(field)
        if isinstance(original, str):
            fixed = repair(original)
            if fixed != original:
                out[field] = fixed
                changed.append(field)

    tags = rec.get("tags")
    if isinstance(tags, list):
        fixed_tags = [repair(t) if isinstance(t, str) else t for t in tags]
        if fixed_tags != tags:
            out["tags"] = fixed_tags
            changed.append("tags")

    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the cleaned file")
    args = ap.parse_args()

    records = load(DATA)
    cleaned = []
    touched = []

    for i, rec in enumerate(records):
        new, changed = clean_record(rec)
        cleaned.append(new)
        if changed:
            touched.append((i, rec, new, changed))

    # ---- refuse to write anything that loses data
    assert len(cleaned) == len(records), "record count changed"
    for old, new in zip(records, cleaned):
        assert bool(old.get("quote", "").strip()) == bool(new.get("quote", "").strip()), \
            "a quote became empty"
        assert len(old.get("tags") or []) == len(new.get("tags") or []), "tag count changed"
        assert set(old.keys()) == set(new.keys()), "fields added or removed"

    print(f"records:        {len(records)}")
    print(f"records fixed:  {len(touched)} ({100 * len(touched) / len(records):.1f}%)")
    print()
    for i, old, new, fields in touched[:6]:
        print(f"  [{i}] {'+'.join(fields)}")
        print(f"      before: {old['quote'][:78]}")
        print(f"      after:  {new['quote'][:78]}")
    if len(touched) > 6:
        print(f"  ... and {len(touched) - 6} more")
    print()

    if not args.apply:
        print("Dry run. Re-run with --apply to write the changes.")
        return 0

    shutil.copy2(DATA, BACKUP)
    print(f"backed up -> {BACKUP.name}")

    with open(DATA, "w", encoding="utf-8") as fh:
        for rec in cleaned:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    reread = load(DATA)
    assert len(reread) == len(records), "re-read count mismatch"
    assert reread == cleaned, "file does not round-trip"
    print(f"wrote {len(reread)} records, verified by re-reading")
    return 0


if __name__ == "__main__":
    sys.exit(main())
