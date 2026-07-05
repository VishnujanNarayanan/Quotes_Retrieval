# Task 2: RAG-Based Semantic Quote Retrieval and Structured QA with Model Training

## 📋 Problem Statement

You are tasked with building a semantic quote retrieval system powered by RAG (Retrieval Augmented Generation) using the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset.

The workflow includes fine-tuning a sentence embedding model, building a retrieval pipeline with FAISS, integrating with a Large Language Model for QA, evaluating the system, and deploying a Streamlit app.

---

## 🛠 Assignment Instructions

### 1. Data Preparation
- Download and explore the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset.
- Preprocess and clean data as needed (tokenization, lowercasing, handling missing values).

### 2. Model Fine-Tuning
- Fine-tune a sentence embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) on quotes.
- Task: Given a query (e.g., "quotes about hope by Oscar Wilde"), retrieve relevant quote, author, and tags.
- Save the fine-tuned model.

### 3. Build the RAG Pipeline
- Encode and index quotes using your fine-tuned model and FAISS.
- Use a Large Language Model (LLM) like GPT or LLaMA to answer natural language queries with retrieved context.

### 4. RAG Evaluation
- Test a set of queries.
- Use any one evaluation framework (RAGAS, Quotient, Arize Phoenix).

### 5. Streamlit Application
- User inputs natural language queries (e.g., “Show me quotes about courage by women authors”).
- System retrieves relevant quotes from the fine-tuned and indexed dataset.
- Structured JSON response with quotes, authors, tags, summary.
- Optionally show source quotes and similarity scores.

### 6. Deliverables
- Jupyter notebook(s) or `.py` files for:
  - Data prep & model fine-tuning
  - RAG pipeline implementation
  - RAG evaluation
  - Streamlit app
- README with instructions, architecture, design, and challenges.
- Demo video walkthrough.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Fine-tune Model

Run your notebook or script to load `quotes.jsonl`, preprocess, and fine-tune the sentence transformer model.

### 3. Run the Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Open the URL shown (usually `http://127.0.0.1:8501`) in your browser.

---

## 🧑‍💻 Example Query and Output

**Input Query:**

```plaintext
quotes about love and bible by C.S. Lewis
```

**Output (retrieved quotes and generated answer):**

```json
{
  "quotes": [
    {
      "quote": "love is the very essence of christianity",
      "author": "c.s. lewis",
      "tags": ["love", "bible", "christianity"],
      "distance": 0.1234
    },
    ...
  ],
  "generated_answer": "Answer generated for 'quotes about love and bible by C.S. Lewis' based on 3 quotes."
}
```

---

## 🧩 Streamlit App Code Snippet

```python
import streamlit as st

st.set_page_config(page_title="Quote Retriever", page_icon="💬")
st.title("💬 Semantic Quote Retriever")

query = st.text_input("Enter your query", placeholder="e.g. Motivational quotes tagged 'accomplishment'")
top_k = st.slider("Number of top quotes", 1, 10, 3)

if st.button("Retrieve Quotes"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            results = retrieve_quotes(query, top_k=top_k)

        if not results:
            st.error("No relevant quotes found.")
        else:
            st.subheader("📜 Retrieved Quotes")
            for res in results:
                st.markdown(f"> {res['quote']}  \n— *{res['author']}*")
                st.caption(f"Tags: {', '.join(res['tags'])} | Distance: {res['distance']:.4f}")

            answer = generate_answer_with_huggingface(query, results)
            st.subheader("🧠 Generated Answer")
            st.success(answer)
```

---

## 📜 License

MIT License — Use and modify freely.





