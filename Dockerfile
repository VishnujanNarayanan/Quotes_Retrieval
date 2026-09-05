# Reproducible runtime for the Streamlit app.
#
# The image ships the fine-tuned model that is committed under
# app/fine-tuned-quote-model/, so the container starts and searches with no
# network access and no training step. Only the LLM synthesis panel needs the
# outside world, and it degrades to "retrieval only" without a token.

FROM python:3.10-slim

# faiss-cpu and torch wheels need libgomp at runtime; the slim image omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies first, so a code change does not re-download ~200 MB of wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# app.py resolves the model directory and the corpus relative to the working
# directory, so the app must start from inside app/ — same as running it locally.
WORKDIR /srv/app

# Build the SQLite database at image build time rather than on first request, so
# a cold container does not pay for it. The app rebuilds it if it is missing.
RUN python build_db.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# headless stops Streamlit trying to open a browser; address 0.0.0.0 makes the
# port reachable from outside the container.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
