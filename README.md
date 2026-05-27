# World Cup 2026 Fantasy Advisor

Streamlit app that recommends the optimal FIFA World Cup 2026 Fantasy squad,
XI, and captain each matchday, projects an expected final-tournament score,
and is validated against 2018 + 2022 historical data.

This is a scaffold. Implementation proceeds in the build-order steps defined
in the project spec; see `TODO(step N)` markers across the codebase for
remaining work.

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your real keys into .env
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud and create a new app pointing at
   `app.py` on the `main` branch.
3. In **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with your real keys filled in.
4. Deploy. Share the resulting URL.

## Required secrets

- `ANTHROPIC_API_KEY` — for the Claude advisor (model: `claude-opus-4-7`).
- `ODDS_API_KEY` — from https://the-odds-api.com/ for tournament + per-player
  goalscorer odds.
