-- Relational form of quotes.jsonl.
--
-- The JSONL file stays the source of truth for the embedding pipeline; this
-- database exists because two of the app's lookups are set operations rather
-- than vector ones. Counting quotes per author and intersecting tag filters were
-- Python loops over the whole corpus on every keystroke; they are one indexed
-- query each here.
--
-- `quotes.rowid_ordinal` is the record's position in quotes.jsonl and therefore
-- its row in the embedding matrix. That correspondence is what lets a SQL filter
-- hand FAISS a candidate set.

DROP TABLE IF EXISTS quote_tags;
DROP TABLE IF EXISTS quotes;

CREATE TABLE quotes (
    ordinal     INTEGER PRIMARY KEY,   -- index into the embedding matrix
    quote       TEXT    NOT NULL,
    author      TEXT    NOT NULL,      -- original casing, trailing comma intact
    author_key  TEXT    NOT NULL       -- trailing comma stripped, for exact matching
);

CREATE TABLE quote_tags (
    ordinal  INTEGER NOT NULL REFERENCES quotes(ordinal) ON DELETE CASCADE,
    tag      TEXT    NOT NULL,         -- lowercased
    PRIMARY KEY (ordinal, tag)
);

CREATE INDEX idx_quotes_author_key ON quotes(author_key);
CREATE INDEX idx_quote_tags_tag    ON quote_tags(tag);
