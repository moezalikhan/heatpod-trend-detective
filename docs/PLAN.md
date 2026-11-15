# HeatPOD Trend Detective: Project Plan

NASA Space Apps Challenge 2026, challenge: **Be an Earth System Trend Detective!** (Advanced, Earth Science and Software)

**Deadline:** project submission closes Sunday, 15 November 2026 at 11:59 PM. For Universal Event teams this follows the local time of the Team Owner's area of residence (Pakistan time). Check the 2026 Project Submission Guide to confirm.

**Stack decision:** the whole project is built in **Python with Streamlit**, hosted on **Streamlit Community Cloud**, with automatic deployment from GitHub.

---

## 1. What we are building

A web app where a student picks a weather variable and a place. It shows 45 years of NASA data (1981 to 2025) and answers three questions in plain words:

1. **Is it changing?** Trend and significance test.
2. **Is it changing the same way everywhere?** POD (Proper Orthogonal Decomposition) pattern map.
3. **Are the worst days getting worse?** Extreme value analysis.

Two ready demos: **Florida** (humid heat) and **Phoenix, Arizona** (dry heat).

---

## 2. How it maps to the challenge objectives

| Challenge objective | Where we cover it |
|---|---|
| Identify variables worth checking | Variable picker with plain definitions (Module 1, Module 8) |
| Get data for a region | NASA POWER regional download (Module 2) |
| Make and show a long time series | 1981 to 2025 series, seasonal cycle removed (Modules 3, 9) |
| Rate of change and significance test | Linear slope with confidence interval, Mann-Kendall test (Module 4) |
| Optional: guided questions and demos | Step by step "investigation" flow and two demos (Module 8) |
| Optional: same process, opposite trends in different regions | POD pattern maps (Module 5) |

---

## 3. Data source: NASA POWER

- **What it is:** NASA's free weather data service (Prediction Of Worldwide Energy Resources). Most of its weather variables come from MERRA-2 (Modern Era Retrospective analysis for Research and Applications, Version 2), the NASA model dataset the challenge recommends.
- **Coverage:** daily data from 1 January 1981 to near real time, about 55 km resolution, no API key needed.
- **Limits to plan around:**
  - Regional requests allow only **one parameter per request**.
  - Maximum area per regional request is about **4.5 x 4.5 degrees**, so Florida needs 4 tiles and Phoenix fits in 1 tile.
  - About **30 requests per 60 seconds**, so the downloader must pause between requests.

### Variables (keep to 4 plus heat index)

| POWER code | Meaning | Plain definition for users |
|---|---|---|
| T2M | Air temperature at 2 meters | How warm the air is at head height |
| T2M_MAX | Daily maximum temperature | The hottest moment of each day |
| RH2M | Relative humidity | How much moisture is in the air, compared with the most it can hold |
| PRECTOTCORR | Precipitation | Rain and snow, in millimeters per day |
| (calculated) Heat index | How hot it feels | Temperature and humidity combined into "feels like" heat |

Heat index uses the NOAA (National Oceanic and Atmospheric Administration) formula with T2M_MAX and RH2M. **Limitation to state honestly:** it combines the daily maximum temperature with daily average humidity, so it is an approximation.

### Regions

- **Florida:** latitude 24.5 to 31.0, longitude minus 87.6 to minus 80.0
- **Phoenix, Arizona:** latitude 31.5 to 35.5, longitude minus 114.5 to minus 110.0 (Sonoran Desert lowlands to the plateau near Flagstaff)
- Ocean grid cells are removed with a land mask (Florida's box contains a lot of sea).

---

## 4. Architecture: precompute, then serve with Streamlit

```
[Python pipeline, runs on the laptop]
NASA POWER API -> download and cache -> clean -> trends / POD / extremes -> results files
                                                                              |
                                                       committed to GitHub    |
                                                                              v
[Streamlit app on Streamlit Community Cloud]   loads results files, draws charts and verdicts
```

**Why this design**

- **The app never calls NASA during judging.** All heavy work (downloads, statistics) is done in advance on the laptop. The app only reads small results files, so it is fast and cannot break because of the NASA API.
- **One language.** Pipeline and app are both Python, in one repository.
- **Small cloud install.** The app needs only light libraries; heavy science libraries stay on the laptop.
- **Fast reruns.** Streamlit reruns the script on every click, so results are loaded once with `st.cache_data` and reused.

---

## 5. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Known well, used for everything |
| Data download | `httpx` with retries | Simple HTTP (web request) client |
| Data handling | `xarray`, `pandas`, `numpy` | xarray handles grid data (latitude x longitude x time) |
| Statistics | `scipy`, `pymannkendall` | `scipy.stats.linregress` for slope, `scipy.stats.genextreme` for GEV, `pymannkendall` for Mann-Kendall |
| POD | `numpy.linalg.svd` | One function call |
| Land mask | `regionmask` with Natural Earth land shapes | Removes ocean cells |
| Results files | Parquet (tables) and JSON (summaries, sentences) | Small, fast to load, easy to inspect |
| Web app | **Streamlit** (version 1.45 or newer, for custom fonts) | Whole UI in Python |
| Charts | **Altair**, shown with `st.altair_chart(chart, theme=None)` | Full control of colors and style; `theme=None` turns off Streamlit's default chart look |
| Styling | `.streamlit/config.toml` theme, plus a small custom CSS file | Gets the "Case File" look without fighting Streamlit |
| Fonts | Self-hosted font files through Streamlit static file serving | No external font loading problems |
| Tests | `pytest`, plus Streamlit `AppTest` | `AppTest` runs the app without a browser and checks it does not crash |
| Code hosting | New public GitHub repository | Linkable on the CV and in the submission |
| CI/CD | GitHub Actions | Runs tests on every push, then releases good code |
| App hosting | **Streamlit Community Cloud** (free) | Connects to GitHub and redeploys automatically on push |

### Two requirements files

- `requirements.txt` (repository root): **only what the app needs** (`streamlit`, `altair`, `pandas`, `numpy`, `pyarrow`, `pyyaml`). Community Cloud installs this, so a short list means fast deploys.
- `pipeline/requirements.txt`: the heavy science tools for the laptop (`xarray`, `netCDF4`, `scipy`, `pymannkendall`, `regionmask`, `httpx`, `pytest`).

---

## 6. Repository setup (from scratch)

We create a **new repository from zero**. No template, no copied starter project.

### Steps

1. On GitHub (account `moezalikhan`), create a new **public** repository named `heatpod-trend-detective`, with no README, no .gitignore, no license (we add them ourselves).
2. On the laptop:
   ```bash
   mkdir heatpod-trend-detective && cd heatpod-trend-detective
   git init -b main
   python -m venv .venv
   ```
3. Create the folder structure (Module 0), then the first commit:
   ```bash
   git add .
   git commit -m "Initial project structure"
   git remote add origin https://github.com/moezalikhan/heatpod-trend-detective.git
   git push -u origin main
   ```
4. Create a second branch called `live` (the branch the public app runs from, see Section 7):
   ```bash
   git push origin main:live
   ```
5. Add a **MIT license** and a README (Module 12).

### Folder structure

```
heatpod-trend-detective/
  .github/
    workflows/
      ci.yml              # tests on every push and pull request
      release.yml         # after tests pass on main, updates the live branch
  .streamlit/
    config.toml           # theme: colors, fonts, static file serving
  static/                 # font files, served by Streamlit
  streamlit_app.py        # app entry point (kept at the root so Streamlit finds config and static files)
  app/
    __init__.py
    data_loader.py        # loads results files with st.cache_data
    styles.css            # small custom CSS (verdict box, lines, spacing)
    views/                # one file per investigation step
      what.py
      where.py
      evidence.py
      patterns.py
      extremes.py
    components/           # reusable pieces
      stripes.py          # warming stripes band
      verdict.py          # verdict box
      charts.py           # Altair chart builders with the project palette
    content/
      glossary.yaml       # plain language definitions for margin notes
  config/
    regions.yaml          # region boxes and tiles
    variables.yaml        # codes, units, plain definitions
  pipeline/
    heatpod/              # download, clean, trends, pod, extremes, export
    tests/
    requirements.txt
  data/
    raw/                  # downloaded NASA files (NOT committed, in .gitignore)
    results/              # output Parquet and JSON (committed, read by the app)
  tests/
    test_app.py           # Streamlit AppTest smoke tests
  requirements.txt        # app only
  CLAUDE.md
  README.md
  LICENSE
  .gitignore
```

---

## 7. Automatic deployment pipeline (CI/CD)

CI/CD means Continuous Integration and Continuous Deployment: every time code is pushed, GitHub automatically checks it and, if the checks pass, the new version of the app goes live.

### The problem we solve

Streamlit Community Cloud redeploys the app whenever the branch it watches gets a new commit. It does **not** wait for tests. If it watched `main` directly, a broken push would break the live app.

### The solution: a `live` branch

- You always push to **`main`**.
- Community Cloud watches a separate branch called **`live`**.
- GitHub Actions copies `main` to `live` **only after all tests pass**.

```
git push to main
      |
      v
GitHub Actions (release.yml)
      |
      +--> Job 1: Pipeline tests   (pytest on pipeline/tests)
      +--> Job 2: App tests        (AppTest: every page loads, no errors, with the real results files)
      |
      v
All passed?
   yes -> Job 3: push main to the live branch
            -> Streamlit Community Cloud sees the new commit on live
            -> app redeploys automatically (usually within a minute or two)
   no  -> live is not touched, the public app keeps the last good version
```

**Key rule:** a broken push never reaches the live app.

### Workflow 1: `ci.yml` (checks on every push and pull request)

- **Triggers:** every push to any branch, and every pull request
- **Steps:** check out code, set up Python 3.11, install both requirements files, run `pytest pipeline/tests` and `pytest tests`
- **Purpose:** catch mistakes early, on any branch

### Workflow 2: `release.yml` (release to live on push to main)

- **Triggers:** push to `main`, plus a manual "Run workflow" button (`workflow_dispatch`)
- **Permissions:** `contents: write` (needed to push to the `live` branch)
- **Concurrency:** one release at a time; a newer push cancels an older in-progress one
- **Jobs:**
  1. `test-pipeline`: run `pytest pipeline/tests`
  2. `test-app`: run `pytest tests` (Streamlit AppTest)
  3. `release` (needs both): `git push origin HEAD:live --force-with-lease`
- **Caching:** cache pip downloads so runs stay fast

### Connecting Streamlit Community Cloud (one time)

1. Go to share.streamlit.io and sign in with GitHub (`moezalikhan`).
2. Click **Create app**, choose **Deploy a public app from GitHub**.
3. Repository: `moezalikhan/heatpod-trend-detective`, branch: **`live`**, main file: `streamlit_app.py`.
4. In **Advanced settings**, choose **Python 3.11**.
5. Pick a custom address, for example `heatpod-trend-detective.streamlit.app`.
6. Deploy. From now on, every successful release updates the app automatically.

If a push ever does not trigger a redeploy, open the app's **Manage app** menu and click **Reboot**.

### Things to know about the free hosting

- **Sleep:** the official docs say all apps without traffic for 12 hours go to sleep. A sleeping app shows a "wake up" button and takes some seconds to start. **Open the app yourself shortly before judging and before any presentation.**
- **Resources are limited**, which is fine because the app only reads small precomputed files.
- **Dependency changes take longer:** if `requirements.txt` changes, Community Cloud does a full reinstall.

### Branch protection (after the hackathon, optional)

In **Settings, Branches**, protect `main` so changes must come through a pull request with a passing `ci.yml`, and protect `live` so only GitHub Actions can push to it.

### Extra: automatic data refresh (after the hackathon)

A scheduled workflow (for example, once a month) could rerun the pipeline, regenerate the results files, and open a pull request with the new data.

---

## 8. Design direction: "The Case File" (Streamlit version)

### Research findings

- **No rainbow color scales.** They can mislead and are unreadable for many colorblind people.
- **Diverging colors for change** (blue for less, red for more, white at zero), as NOAA recommends for anomalies.
- **Grey for "no data" or "not significant."** NASA's Earth Observatory recommends grey so it fades into the background.
- **Intuitive colors:** warm means hotter, blue means cooler.

### Concept

The challenge calls users "data detectives," so the app looks like a scientist's case file or printed research report: **flat colors, thin lines, strong typography. No gradients, no shadows, no glowing banners, no emoji headers, no rounded "blob" cards.**

**Core rule:** the interface stays neutral (paper and ink). **Color is used only for data**, so the eye goes straight to the findings.

### Interface colors

| Use | Color | Hex |
|---|---|---|
| Page background (warm paper) | Off-white | `#F6F3EC` |
| Sidebar and panels | Lighter paper | `#FBF9F4` |
| Main text (ink) | Near-black blue | `#1D2430` |
| Secondary text | Slate | `#5B6472` |
| Divider lines | Warm grey | `#D9D3C7` |
| Interface accent (buttons, links, active step) | Deep ink teal | `#1F4E5F` |
| Not significant / no data | Neutral grey | `#BDBAB3` |

### Data colors (colorblind safe, ColorBrewer)

- **Trend maps, cooling to warming (RdBu):** `#2166AC` `#4393C3` `#92C5DE` `#D1E5F0` `#F7F7F7` `#FDDBC7` `#F4A582` `#D6604D` `#B2182B` (white fixed at zero change)
- **Heat values, low to high (YlOrRd):** `#FFFFCC` `#FED976` `#FD8D3C` `#E31A1C` `#800026`

### Typography

- **Headings:** Source Serif 4 (scientific journal feel)
- **Body:** IBM Plex Sans
- **Numbers, p-values, verdict labels:** IBM Plex Mono (numbers line up and read like evidence)

Font files go in `static/`, are declared in `config.toml` with `[[theme.fontFaces]]`, and are served because `enableStaticServing = true`. Streamlit serves them at the path `app/static/<file name>`.

### Theme file: `.streamlit/config.toml` (outline)

```toml
[server]
enableStaticServing = true

[theme]
base = "light"
primaryColor = "#1F4E5F"
backgroundColor = "#F6F3EC"
secondaryBackgroundColor = "#FBF9F4"
textColor = "#1D2430"
font = "plex-sans"
headingFont = "source-serif"
codeFont = "plex-mono"

[[theme.fontFaces]]
family = "plex-sans"
url = "app/static/IBMPlexSans-Regular.woff2"
weight = 400

# repeat [[theme.fontFaces]] for bold, Source Serif 4 and IBM Plex Mono
```

Claude Code must check the exact theme option names against the installed Streamlit version's documentation before writing this file.

### Custom CSS: `app/styles.css` (keep it small)

Loaded once with `st.markdown(..., unsafe_allow_html=True)`. Only for things the theme cannot do:

- Verdict box: flat 1.5 px ink border, no rounded corners, mono capital label
- Thin divider lines between sections
- Remove shadows and rounded corners from containers
- Hide the default Streamlit footer and menu

### Layout in Streamlit

```
+-----------+----------------------------------------------------+
| SIDEBAR   |  HEATPOD TREND DETECTIVE                           |
|           |  Case: Florida, Heat index                         |
| CASE      |  [Warming stripes band, 1981-2025]                 |
| STEPS     |  +-------------------------------+---------------+ |
|           |  | Main chart                    | MARGIN NOTES  | |
| ( ) 1 What|  | (series + trend line)         |               | |
| ( ) 2 Where| |                               | What is a     | |
| (o) 3 Evid.| | +- VERDICT ----------------+  | p-value?      | |
| ( ) 4 Patt.| | | SIGNIFICANT +0.31 C/dec  |  |               | |
| ( ) 5 Extr.| | | Mann-Kendall p = 0.002   |  | Plain         | |
|           |  | +--------------------------+  | definitions   | |
| [Watch a  |  +-------------------------------+---------------+ |
|  demo]    |                                                    |
+-----------+----------------------------------------------------+
```

- **Sidebar:** the investigation steps as a radio list, plus the region and variable pickers and a "Watch a demo" button.
- **Main area:** `st.columns([3, 1])`. The wide column holds one large chart at a time; the narrow column holds margin notes.
- **Margin notes:** short definitions from `glossary.yaml`, shown next to the chart they explain.
- **Warming stripes band:** an Altair chart of flat colored bars, one per year (blue for cool years, red for warm years). No gradients.
- **Verdict box:** HTML block styled by `styles.css`, with a stamp-style label: `SIGNIFICANT` or `NOT SIGNIFICANT`.
- **Maps:** Altair `mark_rect` grid cells as flat squares, non-significant cells in grey, coastline outline with `mark_geoshape`.
- **Mobile:** Streamlit stacks the columns automatically, so margin notes appear under each chart. The sidebar collapses into a menu.

---

## 9. Modules

Each module is one Claude Code session. Finish and test one before starting the next. **Commit and push to `main` after each module**, so the app updates automatically and there is always a working version online.

### Core (must have today)

**Module 0. Repository, theme, and deployment pipeline**
- Create the repository from scratch (Section 6), including the `live` branch
- Folder structure, `.gitignore`, `CLAUDE.md`, both requirements files, `config/regions.yaml`, `config/variables.yaml`
- `.streamlit/config.toml` with colors and fonts, font files in `static/`
- `ci.yml` and `release.yml` (Section 7)
- Placeholder `streamlit_app.py` showing the title "HeatPOD Trend Detective" in the project fonts and colors
- A first `AppTest` test that checks the placeholder page loads
- Connect Streamlit Community Cloud to the `live` branch
- **Done when:** `streamlit run streamlit_app.py` works locally, the GitHub Actions run is green, and the placeholder page is live on the `streamlit.app` address

**Module 1. Variable catalog**
- The 4 variables plus heat index: code, unit, plain definition, "why it matters" line
- **Done when:** one config file feeds both the pipeline and the app

**Module 2. Data downloader**
- Split each region into tiles of 4.5 degrees or less, and long time ranges into chunks
- Pause between requests (under 30 per minute), retry on failure, cache raw files in `data/raw/`
- Merge tiles into one grid per variable (NetCDF, a standard file format for grid data)
- **Done when:** Florida T2M for 1981 to 2025 loads as one grid with no gaps
- **Tip:** start the Florida downloads early; they take time

**Module 3. Cleaning and preparation**
- Land mask (remove ocean cells)
- Heat index calculation
- **Anomalies:** subtract each month's long-term average to remove the seasonal cycle, as the challenge advises
- Yearly averages and yearly maximums, per grid cell and for the whole region
- Area weighting by the cosine of latitude (grid cells shrink toward the poles)
- **Done when:** a plot of Florida's monthly anomalies shows no summer and winter wave

**Module 4. Trend engine (main challenge requirement)**
- Linear trend: slope per decade with a 95% confidence interval
- **Mann-Kendall test:** checks for a steady rise or fall without assuming a straight line. Use the version that corrects for autocorrelation (when one month's value depends on the previous month's), because the basic test can call random wobbles "significant"
- **Sen's slope:** a robust trend size that outliers do not distort
- Run per grid cell, plus the **Benjamini-Hochberg false discovery rate** correction: when testing 100 cells, a few look significant by pure chance, and this correction prevents that
- Plain language sentence templates, for example: "Heat index has risen by 0.4 C per decade since 1981. This trend is very unlikely to be chance (p = 0.002)."
- **Done when:** fake data with a known trend gives back the right slope, and fake random data with no trend is not called significant

**Module 5. Pattern engine: POD (for Prof. Mou)**
- Build the anomaly matrix (grid cells x months), apply area weights, run SVD (Singular Value Decomposition)
- Keep the modes that explain most of the variation, report the percentage for each
- Mann-Kendall test on each mode's time series (which pattern is trending)
- Per cell trend map showing where trends go opposite ways
- **Done when:** Florida's mode 1 explains a sensible share and its map has a physical meaning (for example, coast vs inland)

**Module 6. Extremes engine: extreme value theory (for Prof. Keellings)**
- Each year's maximum heat index, region wide and per cell
- Fit GEV (Generalized Extreme Value) distributions to 1981 to 2003 and to 2004 to 2025 separately
- Compare the **20 year return level** (heat index expected once every 20 years) between the periods
- Plain words: "The heat level that used to happen once every 20 years now happens about once every N years."
- Bootstrap confidence intervals (refit many times on resampled years to show uncertainty)
- **Done when:** fitting simulated GEV data gives back the parameters used to create it

**Module 7. Export results**
- Write results to `data/results/<region>/<variable>/`: Parquet for series and maps, JSON for summary numbers and plain language sentences
- A schema file describing the results, so pipeline and app always match
- **Done when:** all demo results together are a few MB at most and pass schema checks

**Module 8. App shell and investigation flow**
- Load `styles.css` and set the page layout (`st.set_page_config(layout="wide")`)
- `data_loader.py` with `st.cache_data`
- Sidebar: steps 1 What (variable picker with definitions), 2 Where (Florida, Phoenix), 3 Evidence, 4 Patterns, 5 Extremes
- "Watch a demo" button that sets Florida and heat index and walks through the steps, using `st.session_state` to remember the current step
- **Done when:** all steps are clickable with placeholder charts, live on the app

**Module 9. Charts and views**
- Warming stripes band
- Time series with trend line and confidence band
- Verdict box
- Trend map (grey for not significant)
- POD mode maps with their time series and "% of variation explained"
- Extremes view: return level comparison with uncertainty bars
- All charts built in `components/charts.py` with the project palette and `theme=None`
- **Done when:** every chart reads real results and looks right on a phone width

**Module 10. Explainers**
- Margin notes from `glossary.yaml`: trend, p-value, significance, Mann-Kendall, anomaly, POD, return level
- Simple enough for high school students, as the challenge asks
- **Done when:** a non-scientist can explain the verdict box after reading the notes

**Module 11. Checks**
- Unit tests on fake data (Modules 4 to 6)
- `AppTest` tests: every step loads for both regions and every variable with no exceptions
- Compare one grid cell's series with the official POWER data viewer
- Check on phone and desktop, colorblind simulation, load time after waking from sleep
- **Done when:** all tests pass in GitHub Actions and no chart shows wrong units or labels

**Module 12. Final deploy and submission**
- README: screenshots, live link, method, data source, limitations, how to run the pipeline and the app, license
- Demo video or slides as required by the 2026 Project Submission Guide
- Fill the project page in English, include the live app link and the GitHub link
- Open the app right before submitting so it is awake
- Click **Submit for Judging** with at least 1 hour to spare
- **Done when:** the live link works in a private browser window and the project is submitted

### Extras (only if time is left, or after the hackathon)

- **Module 13. Live custom region:** let users enter any box; the app downloads and analyzes it on demand (slow on free hosting, so show a progress bar and limit the box size)
- **Module 14. Moving trend GEV:** extreme value model where the typical extreme rises smoothly with time, instead of comparing two periods
- **Module 15. Extra variables:** wet bulb temperature, soil moisture from GLDAS (Global Land Data Assimilation System)
- **Module 16. Monthly data refresh workflow** (Section 7)
- **Module 17. React frontend:** rebuild the interface in React for full design control, reusing the same results files

---

## 10. Time plan for today

| Block | Modules | Rough time |
|---|---|---|
| 1 | 0 (repository, theme, CI/CD, Community Cloud), 1, 2 (start Florida downloads early) | 2 hours |
| 2 | 3, 4 | 2 hours |
| 3 | 8 (app shell, while Phoenix downloads) | 1 hour |
| 4 | 5, 6 | 2 hours |
| 5 | 7, 9, 10 | 2.5 hours |
| 6 | 11, 12, submit with at least 1 hour to spare | 1.5 hours |

**If time runs short, cut in this order:** Phoenix demo, bootstrap intervals, per cell extremes (keep region wide only), explainer notes.

**Never cut:** the trend test and verdict box (the challenge's core requirement), and the automatic release pipeline (it guarantees a working live link).

---

## 11. Rules for working with Claude Code (put in `CLAUDE.md`)

- Work on **one module at a time**; only change files that module needs.
- Ask before adding any new dependency, deleting files, or changing the results schema.
- Keep `requirements.txt` (app) small; heavy science libraries go only in `pipeline/requirements.txt`.
- The app must **never** call the NASA API; it only reads `data/results/`.
- Every statistics function gets a test on fake data with a known answer.
- Every new app page gets an `AppTest` check.
- Never commit `data/raw/` or any secrets.
- Run `pytest pipeline/tests`, `pytest tests`, and `streamlit run streamlit_app.py` before saying a module is done.
- Follow the design rules in Section 8: no gradients, no shadows, no emoji, color only for data, all charts with `theme=None`.
- Never push to `live` by hand; only `release.yml` does that.
- After each module: commit with a clear message, push to `main`, and confirm the GitHub Actions run is green and the app updated.

---

## 12. Presentation outline

1. **Problem:** trends are hard to find, and the same variable can trend opposite ways in different places.
2. **Tool:** a guided investigation with tested statistics, explained in plain words.
3. **Two findings:** Florida and Phoenix, one sentence each.
4. **Method rigor:** autocorrelation corrected Mann-Kendall, false discovery rate correction, POD patterns, GEV extremes.
5. **Engineering:** tested release pipeline; only code that passes tests reaches the live app.
6. **Honest limits:** about 55 km grid, approximate heat index, model data rather than weather station data.

---

## Sources

- [Challenge page: Be an Earth System Trend Detective!](https://www.spaceappschallenge.org/2026/challenges/earth-system-trend-detective/)
- [Space Apps 2026 main page (submission times)](https://www.spaceappschallenge.org/2026/)
- [Space Apps Project Submission Guide](https://www.spaceappschallenge.org/resources/project-submission-guide/)
- [NASA POWER Daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- [nasapower R client manual (regional size limit)](https://packages.ropensci.org/nasapower/doc/manual.html)
- [nasapower changelog (rate limit)](https://docs.ropensci.org/nasapower/news/index.html)
- [NASA POWER data evaluation (MERRA-2 source)](https://ntrs.nasa.gov/api/citations/20230015396/downloads/ASA_CSSA_SSSA_Mikovitz.pdf)
- [Streamlit Community Cloud: manage your app (sleep, redeploys)](https://docs.streamlit.io/streamlit-cloud/get-started/manage-your-app)
- [Streamlit: customize fonts](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts)
- [Streamlit: static font files tutorial](https://docs.streamlit.io/develop/tutorials/configuration-and-theming/static-fonts)
- [Streamlit theming overview](https://docs.streamlit.io/en/latest/theme_options.html)
- [NOAA CoastWatch: color choices for data visualization](https://coastwatch.noaa.gov/cwn/news/2021-09-23/colors-and-confusion-making-better-color-choices-data-visualization.html)
- [NASA Earth Observatory: color for data visualization](https://science.nasa.gov/?p=1065218)
- [How rainbow color maps can mislead (Phys.org)](https://phys.org/news/2021-10-rainbow-colour-distort.pdf)
