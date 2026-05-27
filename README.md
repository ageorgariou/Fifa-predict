# World Cup 2026 Fantasy Advisor

A Streamlit web app that picks the best FIFA World Cup 2026 Fantasy squad,
starting XI, and captain for each matchday — and explains the choices in
plain English.

**One URL. One button. Plain-language reasoning.**

The end user is non-technical. The pipeline pulls live data (player prices,
club stats, bookmaker odds, Elo ratings), runs a 10,000-tournament Monte
Carlo, optimizes a budget-constrained 15-man squad via linear programming,
chooses an ownership-aware captain, and asks Claude (Anthropic's Opus model)
to validate against any injury news you paste in.

---

## What it does

Each matchday it returns:

1. **Projected final-tournament score** — calibrated to a realistic
   8-matchday WC range (150 = beginner, 700+ = global winner; advanced/pro
   tier sits at 380-650). Validated against 2018 + 2022 historicals
   (pooled Pearson r = 0.691).
2. **Optimal 15-man squad** — budget $100M / $105M, max 3 per nation,
   2 GK + 5 DEF + 5 MID + 3 FWD.
3. **Best XI for this matchday** — picks the formation (4-4-2, 4-3-3,
   3-4-3 etc.) that maximizes expected points.
4. **Captain + vice** — ownership-aware: Floor mode matches the field,
   Ceiling mode goes contrarian if your league is large enough.
5. **Plain-language strategy + risks** — from Claude, in 2-4 sentences.
6. **Optional**: differentials (low-owned, high-EV picks), chip timing
   recommendation, per-matchday score forecast.

---

## Deploy it (no command line needed once it's on GitHub)

### Step 1 — Get the two API keys (5 minutes)

Both are free for our usage volume.

**Anthropic (for Claude):**
1. Go to https://console.anthropic.com/
2. Sign up / sign in.
3. **Settings → API Keys → Create Key.** Name it `wc-fantasy`.
4. Copy the key (starts with `sk-ant-api03-...`). Paste it somewhere safe
   for now — you won't be able to view it again later.

**The Odds API (for tournament + player goalscorer odds):**
1. Go to https://the-odds-api.com/
2. Sign up (free tier = 500 requests/month — plenty for this app).
3. Copy the API key from your account page.

### Step 2 — Get the code onto GitHub (10 minutes, first time only)

You only need to do this once.

1. Create a free GitHub account at https://github.com if you don't have one.
2. On GitHub, click **+ → New repository**. Name it `wc-fantasy-advisor`.
   Leave it **Private**. Don't add a README / .gitignore (we already have
   them). Click **Create repository**.
3. In your terminal, from inside this project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/wc-fantasy-advisor.git
   git push -u origin main
   ```
   (Replace `YOUR-USERNAME` with your actual GitHub username.)

> **Important:** Make sure `.env` and `.streamlit/secrets.toml` are NOT
> committed. They're in `.gitignore` already, so you should be fine. Double-
> check by going to your GitHub repo in a browser — you should see
> `.env.example` and `.streamlit/secrets.toml.example`, but NOT
> `.env` or `secrets.toml`.

### Step 3 — Deploy on Streamlit Cloud (5 minutes)

1. Go to https://streamlit.io/cloud and sign in with your GitHub account.
2. Click **New app**.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/wc-fantasy-advisor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings → Secrets**, and paste this in (with your
   real keys substituted):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-...your real key..."
   ODDS_API_KEY = "...your real odds-api key..."
   ```
5. Click **Deploy!**.

The first deploy takes 3-5 minutes (installs Python packages + Chrome for
the FBref scraper). After that, your app is live at a URL like
`https://wc-fantasy-advisor-YOUR-USERNAME.streamlit.app`.

Share that URL with anyone who needs access.

### Step 4 — Using it on matchday

For each of the 8 fantasy matchdays:

1. Open the app URL.
2. **Read the yellow checklist at the top.** Do those 4 things 30 minutes
   before kickoff:
   - Check **sofascore.com** for predicted lineups.
   - Skim **@FabrizioRomano, @David_Ornstein** and national-team beat
     reporters on Twitter / X.
   - Paste any injury or rotation news into the **Injuries / news** box
     in the sidebar.
3. Click **Get this round's picks**.
4. Wait ~30-60 seconds. Coffee.
5. Read the projected score, XI, and "Why these picks?" sections.
6. Update your team on play.fifa.com.

---

## Local development (only if you want to tinker)

You only need this if you're changing the code locally. Day-to-day use of
the deployed app doesn't require any of this.

### One-time setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env, paste your real ANTHROPIC_API_KEY and ODDS_API_KEY
```

### Run locally
```bash
.venv/bin/streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run the test suite (sanity check before deploying)
```bash
.venv/bin/python -m model.tournament_sim    # 10k sims in ~2s
.venv/bin/python -m model.projections       # top 30 by ep90
.venv/bin/python -m model.optimizer         # full pipeline
.venv/bin/python -m model.backtest          # 2018 + 2022 backtest
```

### Refresh FBref club stats (only when a new club season ends)
Last-season club stats are cached as CSVs in `data/fbref_cache/`. The
deployed app loads these directly — no `soccerdata` / chromedriver
dependency in production. To rebuild them after a new club season ends:
```bash
.venv/bin/python -m data.fbref_stats refresh
```
This re-scrapes via soccerdata (takes ~2-3 min, needs Chrome installed
locally), writes the new CSVs, then `git commit` + `git push` to ship.

### Update player taker lists between matchdays
Two JSON files control prior multipliers:

- **`data/penalty_takers.json`** — the +1.25× boost for designated PK takers
- **`data/set_piece_takers.json`** — the +1.30× boost for FK / corner takers

Edit them in any text editor when a manager confirms a new PK taker (e.g.
"Bruno Fernandes will take Portugal's penalties this tournament"). Format
is a JSON array of `{"name": ..., "team": ...}` objects.

### Player name mismatch overrides
If a FIFA Fantasy name doesn't match its FBref equivalent (e.g. nicknames,
diacritic variations), add an override to
**`data/name_overrides.json`** in the format:

```json
{"FIFA Name|FIFA Team": "FBref Name"}
```

The match-review file at `data/match_review.csv` lists candidate matches
that scored 85-94 in fuzzy matching — useful for spotting these.

---

## Architecture (at a glance)

```
Raw data sources                       →  Models                    →  UI
─────────────────                        ──────────                    ────
play.fifa.com   (player list + prices)
FBref via       (club-season per-90
  soccerdata     stats)
the-odds-api    (winner + goalscorer
                 markets)
eloratings.net  (team strength)        →  projections.py  →  ep90
                                          ev.py           →  EV per MD
                                          tournament_sim  →  P(qual rounds)
                                          optimizer.py    →  LP 15 squad + XI
                                          captain_opt.    →  captain choice
                                          claude_client   →  advice / risks   →  app.py
                                                                                  (Streamlit)
```

Every module has a `__main__` block — run it directly to inspect its
output.

## Files you might want to edit

| File | What it controls |
|---|---|
| `data/penalty_takers.json` | Players who get the +1.25× PK prior |
| `data/set_piece_takers.json` | Players who get the +1.30× FK prior |
| `data/name_overrides.json` | Manual name-matching fixes |
| `data/manual_players.csv` | Fallback player list if play.fifa.com is down |
| `model/score_estimate.py` (`OFFICE_TIERS`, `GLOBAL_TIERS`) | Score-range tier labels |
| `model/priors.py` (multiplier constants) | Prior strengths if you want to retune |

## Troubleshooting

**"Pipeline failed: invalid x-api-key"** → Your `ANTHROPIC_API_KEY` is wrong
or missing in Streamlit Cloud secrets. Re-paste it, redeploy.

**"No upcoming WC events"** → Bookmaker player markets aren't open yet
(typical 7+ days before MD1). The model falls back to stat-only projections;
nothing's broken.

**"FBref fetch failed"** → Should never happen on the deployed app — last-
season stats are shipped in the repo at `data/fbref_cache/`. If you see
this locally, it means the CSVs are missing; run
`.venv/bin/python -m data.fbref_stats refresh` to rebuild them (needs
Chrome installed locally).

**The projected score looks wrong** → Tier ranges are calibrated for
8-matchday WC fantasy, NOT 38-week Premier League fantasy. A point estimate
of ~435 with "Top 3 finish" / "Advanced player" tier is correct and
healthy — see `model/score_estimate.py` for the canonical tiers.

---

## What this is NOT

- It's not a guarantee. Fantasy is high-variance; a 435 projection has an
  80% confidence range of 370-500, and one bad captain pick can swing
  ±50 points.
- It's not a substitute for reading lineup news. The yellow checklist at
  the top of the app is the most important 30 minutes of your week.
- It's not optimized for win-at-all-costs. Floor mode (default) matches
  the consensus to avoid rank loss; switch to Ceiling mode if you're
  behind in your league and need to swing.
