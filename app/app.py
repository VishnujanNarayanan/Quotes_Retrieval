import html
import os

import streamlit as st

st.set_page_config(
    page_title="Commonplace — Semantic Quote Retrieval",
    page_icon="💬",
    layout="wide",
)
import sys
from pathlib import Path

# Resolve everything relative to this file, not to the working directory. Hosted
# runners (Streamlit Community Cloud, a Hugging Face Space) start the script from
# the repository root, and the previous relative paths only worked when the app
# was launched from inside app/.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import db
import retrieval
from retrieval import RELEVANCE_PCT  # noqa: F401  (documented in Configuration)
from sentence_transformers import SentenceTransformer

# ------------------ Load & Preprocess ------------------

CORPUS = HERE / "quotes.jsonl"
MODEL_DIR = HERE / "fine-tuned-quote-model"

quotes_data = retrieval.load_quotes_data(CORPUS)


@st.cache_resource
def open_db():
    """Connection to quotes.db, built from the corpus if it is not there yet.

    check_same_thread=False because Streamlit reruns the script on a worker
    thread while the cached connection stays put; the app only ever reads.
    """
    if not db.DB_PATH.exists():
        db.build(quotes_data).close()
    return db.connect(check_same_thread=False)


con = open_db()
AUTHORS = db.author_catalogue(con)


@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(str(MODEL_DIR))
    all_quotes = [q['quote'] for q in quotes_data]
    embeddings = model.encode(all_quotes, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().detach().numpy().astype('float32')

    index = retrieval.build_index(embeddings_np)
    return model, index, embeddings_np

model, index, embeddings_np = load_model_and_index()

# ------------------ Quote Retrieval ------------------

def _embed(text):
    vec = model.encode(text, convert_to_tensor=True)
    return vec.cpu().detach().numpy().astype('float32').reshape(1, -1)


def search_quotes(query, author=None, top_k=5):
    """Thin wrapper over retrieval.search_quotes.

    The tag filter is resolved in SQL against quotes.db and handed down as a
    candidate set; the ranking itself is pure and lives in retrieval.py.
    """
    tags = retrieval.parse_advanced_query(query)
    eligible_ids = db.ordinals_with_all_tags(con, tags) if tags else None
    return retrieval.search_quotes(
        query, quotes_data, index, embeddings_np, _embed,
        author=author, top_k=top_k, eligible_ids=eligible_ids,
    )


# ------------------ Answer Generator ------------------

SUMMARY_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

@st.cache_resource
def get_inference_client(hf_token):
    from huggingface_hub import InferenceClient
    return InferenceClient(api_key=hf_token)

def generate_answer_with_huggingface(query, context_quotes, hf_token=None):
    """Summarise the retrieved quotes into a short explanation of the query."""
    if not hf_token:
        return None

    context = "\n".join(
        f"- \"{q['quote']}\" — {q['author']}" for q in context_quotes
    )
    prompt = (
        f"A user searched for: \"{query}\"\n\n"
        f"These quotes were retrieved:\n{context}\n\n"
        "In 3-4 sentences, explain the common theme running through these "
        "quotes and how they answer the search. Refer to the authors by name. "
        "Use only the quotes above — do not invent any. "
        "Write plain prose: no markdown, no bullet points, no headings."
    )

    client = get_inference_client(hf_token)
    response = client.chat_completion(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ------------------ Presentation ------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Spectral:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --qr-ink:    #12161F;
  --qr-leaf:   #1B2230;
  --qr-vellum: #ECE7DC;
  --qr-brass:  #C8A24A;
  --qr-sage:   #7E9B8A;
  --qr-muted:  #8A93A5;
}

.stApp { background: var(--qr-ink); }
.block-container { padding-top: 2.6rem; max-width: 1320px; }

/* ---- masthead ---- */
.qr-masthead { margin-bottom: 2rem; }
.qr-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem; letter-spacing: .22em; text-transform: uppercase;
  color: var(--qr-brass); margin-bottom: .55rem;
}
.qr-title {
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(2.1rem, 4.2vw, 3.1rem); font-weight: 600;
  color: var(--qr-vellum); line-height: 1.04; margin: 0 0 .5rem 0;
}
.qr-standfirst {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  font-size: .95rem; color: var(--qr-muted);
  max-width: 62ch; line-height: 1.55; margin: 0;
}
.qr-rule { height: 1px; background: linear-gradient(90deg, var(--qr-brass), transparent 62%); margin: 1.5rem 0 0 0; }

/* ---- column headers ---- */
.qr-colhead {
  display: flex; align-items: baseline; gap: .7rem;
  border-bottom: 1px solid rgba(200,162,74,.28);
  padding-bottom: .55rem; margin-bottom: 1.4rem;
}
.qr-colhead h2 {
  font-family: 'Fraunces', Georgia, serif; font-size: 1.22rem; font-weight: 600;
  color: var(--qr-vellum); margin: 0; letter-spacing: .01em;
}
.qr-colhead span {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem;
  color: var(--qr-muted); letter-spacing: .08em; text-transform: uppercase;
}

/* ---- quote card ---- */
.qr-card {
  background: var(--qr-leaf);
  border: 1px solid rgba(236,231,220,.07);
  border-left: 2px solid var(--qr-brass);
  border-radius: 3px;
  padding: 1.5rem 1.7rem 1.25rem;
  margin-bottom: 1.1rem;
  animation: qr-rise .5s cubic-bezier(.22,.7,.3,1) both;
}
.qr-rank {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem;
  color: var(--qr-brass); letter-spacing: .14em; margin-bottom: .7rem;
}
.qr-quote {
  font-family: 'Spectral', Georgia, serif;
  font-size: 1.18rem; line-height: 1.72; color: var(--qr-vellum);
  margin: 0 0 .95rem 0; font-weight: 400;
}
.qr-author {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  font-size: .82rem; font-weight: 600; letter-spacing: .05em;
  text-transform: uppercase; color: var(--qr-sage); margin-bottom: 1rem;
}
.qr-tags { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.1rem; }
.qr-tag {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
  color: var(--qr-sage); border: 1px solid rgba(126,155,138,.3);
  border-radius: 2px; padding: .16rem .5rem;
}

/* ---- fillers: present, but visibly secondary ---- */
.qr-card.qr-filler { border-left-color: var(--qr-sage); background: rgba(27,34,48,.55); }
.qr-card.qr-filler .qr-rank { color: var(--qr-sage); }
.qr-card.qr-filler .qr-quote { font-size: 1.06rem; color: rgba(236,231,220,.82); }
.qr-fillhead {
  display: flex; align-items: center; gap: .8rem;
  margin: 1.9rem 0 1.1rem;
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
  letter-spacing: .14em; text-transform: uppercase; color: var(--qr-sage);
}
.qr-fillhead::after { content: ""; flex: 1; height: 1px; background: rgba(126,155,138,.22); }

/* ---- the gilt similarity rule: the signature ---- */
.qr-meter { display: flex; align-items: center; gap: .8rem; }
.qr-track { flex: 1; height: 2px; background: rgba(236,231,220,.09); position: relative; }
.qr-fill  { height: 2px; background: linear-gradient(90deg, var(--qr-brass), rgba(200,162,74,.42)); }
.qr-dist {
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem;
  color: var(--qr-muted); white-space: nowrap;
}

/* ---- synthesis panel ---- */
.qr-synth {
  background: linear-gradient(160deg, rgba(200,162,74,.06), rgba(27,34,48,.9));
  border: 1px solid rgba(200,162,74,.22);
  border-radius: 3px; padding: 1.7rem 1.8rem;
  position: sticky; top: 1rem;
}
.qr-synth p {
  font-family: 'Spectral', Georgia, serif;
  font-size: 1.06rem; line-height: 1.78; color: var(--qr-vellum);
  margin: 0 0 1rem 0;
}
.qr-synth p:last-child { margin-bottom: 0; }
.qr-attrib {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
  color: var(--qr-muted); line-height: 1.6;
  border-top: 1px solid rgba(236,231,220,.1);
  margin-top: 1.3rem; padding-top: .85rem;
}
.qr-attrib b { color: var(--qr-brass); font-weight: 500; }

/* ---- empty + notice states ---- */
.qr-empty {
  border: 1px dashed rgba(236,231,220,.14); border-radius: 3px;
  padding: 2.2rem 1.9rem; text-align: center;
}
.qr-empty p {
  font-family: 'Spectral', Georgia, serif; font-style: italic;
  color: var(--qr-muted); font-size: 1rem; margin: 0 0 1.1rem 0;
}
.qr-examples { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; }
.qr-ex {
  font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
  color: var(--qr-vellum); background: rgba(236,231,220,.05);
  border: 1px solid rgba(236,231,220,.12); border-radius: 2px; padding: .3rem .65rem;
}

@keyframes qr-rise { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .qr-card { animation: none; } }
@media (max-width: 900px) { .qr-synth { position: static; } }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def closeness(distance):
    """Map an L2 distance onto a 0-1 scale for the gilt rule."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 1.5))


def quote_card(rank, res):
    tags = "".join(
        f'<span class="qr-tag">{html.escape(t)}</span>' for t in res["tags"]
    )
    filler = " qr-filler" if res.get("is_filler") else ""
    if res["distance"] is None:
        meter = '<div class="qr-meter"><div class="qr-track"></div>' \
                '<div class="qr-dist">from the anthology</div></div>'
    else:
        pct = closeness(res["distance"]) * 100
        meter = (
            f'<div class="qr-meter">'
            f'<div class="qr-track"><div class="qr-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="qr-dist">L2 {res["distance"]:.4f}</div></div>'
        )
    return f"""
    <div class="qr-card{filler}" style="animation-delay:{rank * 0.06:.2f}s">
      <div class="qr-rank">{rank:02d}</div>
      <p class="qr-quote">{html.escape(res['quote'])}</p>
      <div class="qr-author">{html.escape(res['author'].rstrip(','))}</div>
      <div class="qr-tags">{tags}</div>
      {meter}
    </div>
    """


def column_head(title, note):
    return f'<div class="qr-colhead"><h2>{title}</h2><span>{note}</span></div>'


# ------------------ Streamlit UI ------------------

st.markdown(
    f"""
    <div class="qr-masthead">
      <div class="qr-eyebrow">Semantic retrieval · {len(quotes_data):,} quotes</div>
      <h1 class="qr-title">Commonplace</h1>
      <p class="qr-standfirst">Search a literary anthology by meaning rather than wording.
      A fine-tuned sentence-transformer embeds every quote; FAISS ranks them against your
      query; a language model reads the results back to you.</p>
      <div class="qr-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

search_col, author_col, count_col = st.columns([2.2, 1.5, 1.1])
with search_col:
    query = st.text_input(
        "Search by meaning",
        placeholder="Enter a search term, or leave blank to browse an author",
    )
with author_col:
    author_choice = st.selectbox(
        "Author",
        ["Any author"] + [f"{name} ({n})" for name, n in AUTHORS],
        help="Anchors results to one author. Remaining slots are filled with "
             "related quotes from others.",
    )
    author = None if author_choice == "Any author" else author_choice.rsplit(" (", 1)[0]
with count_col:
    top_k = st.slider("Quotes to retrieve", 1, 10, 5)

def _saved_token():
    """Token from .streamlit/secrets.toml, falling back to the HF_TOKEN env var."""
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        token = ""
    if not token or token.startswith("PASTE_YOUR"):
        token = os.environ.get("HF_TOKEN", "")
    return token

hf_token = _saved_token()
if not hf_token:
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        help="Create a free token at huggingface.co/settings/tokens "
             "(Fine-grained -> Inference preset), or save it in "
             "app/.streamlit/secrets.toml to skip this field.",
    )

search = st.button("Retrieve & read", type="primary")

if search and not query.strip() and not author:
    st.warning("Enter a search term, pick an author, or both.")

elif search:
    with st.spinner("Embedding query and searching the index..."):
        results = search_quotes(query, author=author, top_k=top_k)

    if not results:
        st.error("Nothing matched those filters. Try a broader tag or a different author.")
    else:
        primary = [r for r in results if not r["is_filler"]]
        fillers = [r for r in results if r["is_filler"]]

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            if author:
                note = f"{len(primary)} by {author} · {len(fillers)} related" if fillers \
                    else f"{len(primary)} by {author}"
            else:
                note = f"{len(results)} of {len(quotes_data):,} · ranked by distance"
            st.markdown(column_head("Retrieved", note), unsafe_allow_html=True)

            for i, res in enumerate(primary, start=1):
                st.markdown(quote_card(i, res), unsafe_allow_html=True)

            if author and not primary:
                st.info(
                    f"No quotes by {author} are close enough to “{query.strip()}” "
                    f"to count as a match. Showing the closest quotes from other authors."
                )

            if fillers:
                label = "Related, from other authors" if primary else "Closest matches"
                st.markdown(
                    f'<div class="qr-fillhead">{label}</div>',
                    unsafe_allow_html=True,
                )
                for i, res in enumerate(fillers, start=len(primary) + 1):
                    st.markdown(quote_card(i, res), unsafe_allow_html=True)

        with right:
            st.markdown(
                column_head("Synthesis", "Hugging Face"),
                unsafe_allow_html=True,
            )
            if not hf_token:
                st.markdown(
                    '<div class="qr-empty"><p>Add a Hugging Face token to have '
                    'these quotes read back as a single passage.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    with st.spinner(f"{SUMMARY_MODEL.split('/')[-1]} is reading the results..."):
                        answer = generate_answer_with_huggingface(query, results, hf_token)
                    paragraphs = "".join(
                        f"<p>{html.escape(p.strip())}</p>"
                        for p in answer.split("\n") if p.strip()
                    )
                    st.markdown(
                        f"""
                        <div class="qr-synth">
                          {paragraphs}
                          <div class="qr-attrib">
                            Generated by <b>{html.escape(SUMMARY_MODEL)}</b><br>
                            via the Hugging Face Inference API, grounded only in
                            the {len(results)} retrieved quotes.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")

else:
    st.markdown(
        """
        <div class="qr-empty">
          <p>"The best of a book is not the thought which it contains,
          but the thought which it suggests."</p>
          <div class="qr-examples">
            <span class="qr-ex">courage in the face of fear</span>
            <span class="qr-ex">wisdom by mark twain</span>
            <span class="qr-ex">tagged with both 'love' and 'life'</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
