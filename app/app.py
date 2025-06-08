import streamlit as st
st.set_page_config(page_title="Quote Retriever", page_icon="💬")
import pandas as pd
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------ Load & Preprocess ------------------

def load_quotes_data():
    df = pd.read_json("quotes.jsonl", lines=True)
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
                'tags_lc': [t.lower() for t in raw_tags]
            })
    return clean_data
quotes_data = load_quotes_data()

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("fine-tuned-quote-model")
    all_quotes = [q['quote'] for q in quotes_data]
    embeddings = model.encode(all_quotes, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().detach().numpy().astype('float32')

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return model, index, embeddings_np

model, index, embeddings_np = load_model_and_index()

# ------------------ Query Parsing ------------------

def parse_advanced_query(query):
    tags = []
    author = None
    tag_match = re.search(r"tagged with both ['\"]?(\w+)['\"]?\s+and\s+['\"]?(\w+)", query, re.IGNORECASE)
    if tag_match:
        tags = [tag_match.group(1).lower(), tag_match.group(2).lower()]
    else:
        tag_match = re.search(r"tagged ['\"]?(\w+)['\"]?", query, re.IGNORECASE)
        if tag_match:
            tags = [tag_match.group(1).lower()]

    author_match = re.search(r"by ([\w\s.]+)", query, re.IGNORECASE)
    if author_match:
        author = author_match.group(1).strip().lower()

    return tags, author

# ------------------ Quote Retrieval ------------------

def retrieve_quotes(query, top_k=3):
    filter_tags, filter_author = parse_advanced_query(query)

    filtered_quotes = []
    filtered_embeddings = []
    for i, q in enumerate(quotes_data):
        q_tags = q['tags_lc']
        q_author = q['author_lc']


        tag_match = all(tag in q_tags for tag in filter_tags) if filter_tags else True
        author_match = (filter_author in q_author) if filter_author else True

        if tag_match and author_match:
            filtered_quotes.append(q)
            filtered_embeddings.append(embeddings_np[i])

    if not filtered_quotes:
        return []

    clean_query = re.sub(r"tagged with.*|tagged.*|by .*", "", query, flags=re.IGNORECASE).strip()
    query_embedding = model.encode(clean_query, convert_to_tensor=True)
    query_np = query_embedding.cpu().detach().numpy().astype('float32').reshape(1, -1)

    temp_index = faiss.IndexFlatL2(query_np.shape[1])
    temp_index.add(np.array(filtered_embeddings).astype('float32'))

    distances, indices = temp_index.search(query_np, min(top_k, len(filtered_quotes)))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        q = filtered_quotes[idx]
        results.append({
            'quote': q['quote'],
            'author': q['author'],
            'tags': q['tags'],
            'distance': dist
        })
    return results

# ------------------ Placeholder Answer Generator ------------------

def generate_answer_with_huggingface(query, context_quotes, hf_token=None):
    return f"Answer for: '{query}'\n\n" + "\n".join([f"• {q['quote']} — {q['author']}" for q in context_quotes])

# ------------------ Streamlit UI ------------------


st.title("💬 Semantic Quote Retriever")

query = st.text_input("Enter your query", placeholder="e.g. quotes about happiness tagged 'buddha' by 'dalai lama'")
top_k = st.slider("Number of top quotes to retrieve", 1, 10, 3)

hf_token = st.text_input("HuggingFace Token (optional)", type="password")

if st.button("Retrieve & Generate"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching and generating..."):
            results = retrieve_quotes(query, top_k=top_k)

            if not results:
                st.error("No relevant quotes found.")
            else:
                st.subheader("📜 Retrieved Quotes")
                for res in results:
                    st.markdown(f"> *{res['quote']}*  \n— **{res['author']}**")
                    st.caption(f"Tags: {', '.join(res['tags'])} | Distance: {res['distance']:.4f}")

                st.subheader("🧠 Generated Answer")
                answer = generate_answer_with_huggingface(query, results, hf_token)
                st.success(answer)
