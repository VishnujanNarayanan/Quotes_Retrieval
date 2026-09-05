-- Named queries, read at runtime by db.py.
--
-- They live in SQL rather than in Python string literals so they can be run
-- against quotes.db by hand (sqlite3 quotes.db < queries.sql won't work — these
-- are parameterised — but each one pastes straight into a sqlite3 prompt).
--
-- Format: each query starts with a `-- name: <key>` line. db.py splits on those.

-- name: author_catalogue
-- Authors ordered by how many quotes they have, for the picker. Ordering by
-- count then name keeps the dropdown stable between runs.
SELECT author_key, COUNT(*) AS n
FROM quotes
WHERE author_key <> ''
GROUP BY author_key
ORDER BY n DESC, author_key ASC;

-- name: ordinals_with_all_tags
-- The candidate set for a tag filter: rows carrying EVERY tag passed in, which
-- is what "tagged with both 'love' and 'life'" asks for. HAVING COUNT = the
-- number of requested tags is the intersection; a WHERE tag IN (...) alone would
-- be a union and would quietly widen the filter.
SELECT ordinal
FROM quote_tags
WHERE tag IN (%s)
GROUP BY ordinal
HAVING COUNT(DISTINCT tag) = ?
ORDER BY ordinal;

-- name: quotes_by_author
-- Every ordinal belonging to one author, in corpus order.
SELECT ordinal
FROM quotes
WHERE author_key = ?
ORDER BY ordinal;

-- name: tag_frequencies
-- How often each tag is used across the anthology. Not on the search path; it is
-- what a maintainer runs to see whether a tag filter is worth offering at all.
SELECT tag, COUNT(*) AS n
FROM quote_tags
GROUP BY tag
ORDER BY n DESC, tag ASC;

-- name: corpus_summary
-- One row describing the whole corpus, used by build_db.py to report what it wrote.
SELECT
    (SELECT COUNT(*)                 FROM quotes)     AS quotes,
    (SELECT COUNT(DISTINCT author_key) FROM quotes)   AS authors,
    (SELECT COUNT(DISTINCT tag)      FROM quote_tags) AS tags;
