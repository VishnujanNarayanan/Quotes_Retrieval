"""Retrieval logic, independent of Streamlit and of the embedding model.

app.py owns the UI, the caching and the model; this module owns the search. The
split exists so the ranking rules can be tested without loading a 91 MB
sentence-transformer or starting a Streamlit session — everything here works
against a plain float32 array and a callable that turns text into a vector.
"""

import re

import faiss
import numpy as np
import pandas as pd

# Distances shift per query (best match for "courage in the face of fear" is 0.39,
# for "love" it is 0.99), so a fixed cutoff cannot work. A quote counts as on-topic
# if it lands in the closest RELEVANCE_PCT of the corpus for that particular query.
RELEVANCE_PCT = 5.0

_TAG_PHRASE = re.compile(r"tagged with.*|tagged.*", re.IGNORECASE)


def load_quotes_data(path):
    """Read the JSONL corpus into the record shape the rest of the app expects."""
    df = pd.read_json(path, lines=True)
    clean_data = []
    for _, row in df.iterrows():
        raw_quote = str(row.get('quote', '')).strip()
        raw_author = str(row.get('author', 'unknown')).strip()
        raw_tags = [tag.strip() for tag in row.get('tags', []) if tag.strip()]

        if raw_quote:
            clean_data.append({
                'quote': raw_quote,         # ← keep original casing
                'author': raw_author,       # ← keep original casing
                'tags': raw_tags,           # ← keep original casing
                'quote_lc': raw_quote.lower(),   # for searching/filtering
                'author_lc': raw_author.lower(),
                # trailing commas are an artefact of the scrape; strip for exact matching
                'author_key': raw_author.rstrip(',').strip(),
                'tags_lc': [t.lower() for t in raw_tags]
            })
    return clean_data


def parse_advanced_query(query):
    """Pull tag filters out of the query text. Author has its own field."""
    tags = []
    tag_match = re.search(
        r"tagged with both ['\"]?(\w+)['\"]?\s+and\s+['\"]?(\w+)", query, re.IGNORECASE
    )
    if tag_match:
        tags = [tag_match.group(1).lower(), tag_match.group(2).lower()]
    else:
        tag_match = re.search(r"tagged ['\"]?(\w+)['\"]?", query, re.IGNORECASE)
        if tag_match:
            tags = [tag_match.group(1).lower()]

    return tags


def strip_tag_phrases(query):
    """The text left once the tag filter is removed, so the filter and the
    semantic query do not interfere with each other."""
    return _TAG_PHRASE.sub("", query).strip()


def build_index(embeddings_np):
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index


def _pack(quotes_data, idx, distance, is_filler):
    q = quotes_data[idx]
    return {
        'quote': q['quote'],
        'author': q['author_key'] or q['author'],
        'tags': q['tags'],
        'distance': None if distance is None else float(distance),
        'is_filler': is_filler,
    }


def search_quotes(query, quotes_data, index, embeddings_np, embed, author=None,
                  top_k=5, eligible_ids=None):
    """Rank quotes by meaning, optionally anchored to one author.

    Author quotes that are genuinely on-topic come first. Whatever slots remain —
    because the author has too few quotes, or their remaining ones drift off-topic —
    are filled with the closest quotes from other authors, flagged as fillers.

    `embed` turns a string into a (1, dim) float32 array. `eligible_ids`, when
    given, is the pre-filtered candidate set (the SQL tag filter supplies it);
    otherwise the tag filter is applied in Python over `quotes_data`.
    """
    filter_tags = parse_advanced_query(query)
    clean_query = strip_tag_phrases(query)

    if eligible_ids is None:
        def tags_ok(i):
            if not filter_tags:
                return True
            return all(t in quotes_data[i]['tags_lc'] for t in filter_tags)
    else:
        allowed = set(eligible_ids)
        def tags_ok(i):
            return i in allowed

    def by_author(i):
        return bool(author) and quotes_data[i]['author_key'] == author

    eligible = [i for i in range(len(quotes_data)) if tags_ok(i)]
    if not eligible:
        return []

    # ---- author only: no query to rank by, so anchor on what the author writes about
    if not clean_query:
        if not author:
            return []
        own = [i for i in eligible if by_author(i)]
        # no query means no distance to report — the meter would be meaningless
        results = [_pack(quotes_data, i, None, False) for i in own[:top_k]]
        if len(results) < top_k and own:
            centroid = embeddings_np[own].mean(axis=0).reshape(1, -1)
            dist, idx = index.search(centroid, len(quotes_data))
            own_set = set(own)
            for d, i in zip(dist[0], idx[0], strict=True):
                if len(results) >= top_k:
                    break
                if i not in own_set and tags_ok(i):
                    results.append(_pack(quotes_data, i, d, True))
        return results

    # ---- one FAISS pass over the whole corpus gives ranking and threshold together
    query_np = embed(clean_query)
    dist, idx = index.search(query_np, len(quotes_data))
    ranked = [(d, i) for d, i in zip(dist[0], idx[0], strict=True) if tags_ok(i)]

    if not author:
        return [_pack(quotes_data, i, d, False) for d, i in ranked[:top_k]]

    cutoff = np.percentile([d for d, _ in ranked], RELEVANCE_PCT)

    results = [
        _pack(quotes_data, i, d, False) for d, i in ranked
        if by_author(i) and d <= cutoff
    ][:top_k]

    if len(results) < top_k:
        for d, i in ranked:
            if len(results) >= top_k:
                break
            if not by_author(i):
                results.append(_pack(quotes_data, i, d, True))

    return results
