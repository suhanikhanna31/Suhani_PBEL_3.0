# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn/xgboost wheels + nltk build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora at build time so the running container never
# needs outbound network access to nltk servers (Code Engine egress is
# otherwise fine, but this makes cold starts faster and builds reproducible).
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in \
    ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"

COPY . .

# Code Engine injects PORT; default to 8000 for local docker run.
ENV PORT=8000
EXPOSE 8000

# Run the pipeline once at container start (populates data/processed/ from
# whatever's in data/raw/ — mount/COPY real CERT data in before deploying,
# or the synthetic generator fills in for a live demo) then start the API.
CMD sh -c "python -m src.pipeline; uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"
