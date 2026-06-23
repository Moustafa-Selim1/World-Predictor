"""
Predictive Type — Railway deployment entrypoint
Serves the next-word prediction API AND the static frontend from a single
process, listening on the port Railway provides via the PORT env var.
"""

import os
import pickle
import re
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Predictive Type API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH} ...")
with open(MODEL_PATH, "rb") as f:
    MODEL, N = pickle.load(f)
print(f"Model loaded. Order n={N}, states={len(MODEL):,}")

WORD_RE = re.compile(r"[A-Za-z']+")
WORD_ONLY_RE = re.compile(r"^[A-Za-z']+$")


def tokenize(text: str):
    """Lowercase word tokens, matching how the model was almost certainly trained."""
    return WORD_RE.findall(text.lower())


def _is_word(token: str) -> bool:
    """True for real alphabetic word tokens; filters out '*' padding and punctuation."""
    return bool(WORD_ONLY_RE.match(token))


def predict_next(text: str, top_k: int = 8):
    """
    Look up the last N words as context and return the top_k most frequent
    next-words, backing off to shorter context (then sentence-starters)
    if the full context was never seen during training.

    Each state in MODEL maps to a {word: count} dict (pre-sorted-able by
    count) rather than a raw list of repeated words -- this is a compact
    representation that keeps only the top candidates per state to save
    space, produced by a one-time offline pruning pass over the original
    training data.
    """
    tokens = tokenize(text)

    for order in range(N, 0, -1):
        if len(tokens) < order:
            continue
        key = tuple(tokens[-order:]) if order > 1 else (tokens[-1],)
        if order < N:
            key = tuple(["*"] * (N - order)) + key
        candidates = MODEL.get(key)
        if candidates:
            ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
            words = [w for w, _ in ranked if _is_word(w)][:top_k]
            if words:
                return words

    start_key = tuple(["*"] * N)
    candidates = MODEL.get(start_key)
    if candidates:
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
        return [w for w, _ in ranked if _is_word(w)][:top_k]

    return []


# ---------------------------------------------------------------------------
# API routes (registered before static mount so they take priority)
# ---------------------------------------------------------------------------
@app.get("/api/predict")
def predict(text: str = Query(default=""), top_k: int = Query(default=8, ge=1, le=20)):
    suggestions = predict_next(text, top_k=top_k)
    return {"text": text, "suggestions": suggestions}


@app.get("/api/health")
def health():
    return {"status": "ok", "states": len(MODEL), "order": N}


# ---------------------------------------------------------------------------
# Static frontend (the departure-board UI)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
