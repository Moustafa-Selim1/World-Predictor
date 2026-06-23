# Predictive Type

A next-word predictor (trigram language model) with a departure-board themed
UI, deployable to Railway (railway.app) straight from GitHub.

## Project structure

```
.
├── app.py              FastAPI app — serves the API AND the static frontend
├── model.pkl            compact trigram model (~22MB — pushes to GitHub normally)
├── compact_model.py     script used to shrink the original 240MB model to this size
├── static/
│   └── index.html      the frontend UI (single file, no build step)
├── requirements.txt    Python dependencies
├── railway.json         Railway build/start config
├── Procfile             fallback start command
└── .gitignore
```

One process serves everything: `/api/predict` and `/api/health` are the API
routes, and everything else falls through to the static files in `static/`.

## About model.pkl's size

Your original trained model was ~240MB. That's well over GitHub's 100MB
per-file limit, so it would have needed Git LFS to push at all.

Instead, `model.pkl` here has been compacted to about **22MB** using
`compact_model.py`, which:
- collapses each state's repeated word list into a `{word: count}` dict
  (no duplicate storage of the same word)
- drops punctuation-only tokens (the UI never shows those anyway)
- drops states seen fewer than 7 times total (too sparse to usefully rank)
- keeps only the top 8 most frequent next-words per state (the UI never
  shows more than 8 suggestions anyway)

This was checked against the original model and produces **identical top-8
predictions** for common phrases — only extremely rare, low-confidence
single-occurrence phrases are affected, and the app already falls back
gracefully to shorter context for those.

Because it's under 100MB, **no Git LFS is required** — a normal
`git add` / `git push` just works.

## Step-by-step: push to GitHub

```bash
cd predictive-type        # the folder containing app.py, static/, etc.
git init
git add .
git commit -m "Initial commit: predictive type app"
```

Create a new GitHub repo (on github.com, click "New repository" — don't
initialize it with a README since you already have files locally), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step-by-step: deploy on Railway

1. Go to https://railway.app and sign in (GitHub login is easiest).
2. Click **New Project → Deploy from GitHub repo**.
3. Select the repo you just pushed.
4. Railway detects it's a Python app via `requirements.txt` and uses the
   `startCommand` from `railway.json` (`python app.py`) automatically.
5. Once deployed, go to **Settings → Networking** and click **Generate
   Domain** to get a public `*.up.railway.app` URL.
6. Open that URL — you should see the departure-board UI, and the status
   line should read **ONLINE**.

## Local testing before you deploy

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:8000** in your browser.

## How the model works (recap)

- It's a **trigram model**: it looks at the last 2 words you typed and
  predicts the next one based on what followed that pair in training
  (Project Gutenberg books).
- If your exact 2-word phrase was never seen often enough in training, it
  automatically falls back to 1-word context, then to general
  sentence-starter words.
- Suggestions are filtered to real words only — punctuation tokens the
  model learned internally are hidden from the UI.

## If you ever want to re-run the compaction yourself

If you retrain a bigger/different model later and need to shrink it again:

```bash
python compact_model.py original_model.pkl model.pkl
```

Tune `MIN_TOTAL_OBSERVATIONS` and `TOP_K_PER_STATE` at the top of that file
to trade off size vs. coverage of rarer phrases.
