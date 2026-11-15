# CLAUDE.md

## Project summary

HeatPOD Trend Detective is a NASA Space Apps 2026 entry for the challenge
"Be an Earth System Trend Detective!". It is a Streamlit web app where a student
picks a weather variable (T2M, T2M_MAX, RH2M, PRECTOTCORR, or calculated heat
index) and a region (Florida or Punjab, Pakistan) and sees 45 years of NASA POWER
data (1981 to 2025). It answers three questions in plain words:

1. Is it changing? (linear trend, Mann-Kendall test)
2. Is it changing the same way everywhere? (POD pattern maps)
3. Are the worst days getting worse? (extreme value analysis, GEV)

Architecture: a Python pipeline runs on the laptop, downloads NASA POWER data,
computes all statistics, and writes small results files to `data/results/`.
The Streamlit app only reads those files. It is hosted on Streamlit Community
Cloud from the `live` branch, which GitHub Actions updates only after tests pass.

`docs/PLAN.md` is the single source of truth. Read it before starting any module.

## Folder map

```
heatpod-trend-detective/
  .github/workflows/
    ci.yml              # tests on every push and pull request
    release.yml         # after tests pass on main, updates the live branch
  .streamlit/
    config.toml         # theme: colors, fonts, static file serving
  static/               # font files, served by Streamlit at app/static/<file>
  streamlit_app.py      # app entry point (root, so Streamlit finds config and static files)
  app/
    data_loader.py      # loads results files with st.cache_data
    styles.css          # small custom CSS (verdict box, lines, spacing)
    views/              # one file per investigation step
    components/         # reusable pieces (stripes, verdict, charts)
    content/            # glossary.yaml with plain language definitions
  config/
    regions.yaml        # region boxes and tiles
    variables.yaml      # codes, units, plain definitions
  pipeline/
    heatpod/            # download, clean, trends, pod, extremes, export
    tests/
    requirements.txt    # heavy science tools (laptop only)
  data/
    raw/                # downloaded NASA files (NOT committed)
    results/            # output Parquet and JSON (committed, read by the app)
  tests/
    test_app.py         # Streamlit AppTest smoke tests
  requirements.txt      # app only
```

## Rules

- Work on one module at a time; only change files that module needs.
- Ask before adding any new dependency, deleting files, or changing the results schema.
- Keep `requirements.txt` (app) small; heavy science libraries go only in `pipeline/requirements.txt`.
- The app must never call the NASA API; it only reads `data/results/`.
- Every statistics function gets a test on fake data with a known answer.
- Every new app page gets an `AppTest` check.
- Never commit `data/raw/` or any secrets.
- Run `pytest pipeline/tests`, `pytest tests`, and `streamlit run streamlit_app.py` before saying a module is done.
- Follow the design rules in docs/PLAN.md Section 8: no gradients, no shadows, no emoji, color only for data, all charts with `theme=None`.
- Never push to `live` by hand; only `release.yml` does that.
- After each module: commit with a clear message, push to `main`, and confirm the GitHub Actions run is green and the app updated.

## Local commands

```bash
source .venv/bin/activate
pytest pipeline/tests
pytest tests
streamlit run streamlit_app.py
```
