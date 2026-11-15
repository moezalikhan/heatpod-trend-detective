# Space Apps 2026 submission notes (draft to paste into the project page)

**Project name:** HeatPOD Trend Detective
**Challenge:** Be an Earth System Trend Detective!
**Live app:** https://heatpod-trend-detective.streamlit.app/
**Repository:** https://github.com/moezalikhan/heatpod-trend-detective

## High-level summary (short)

HeatPOD Trend Detective is a web app that turns 45 years of NASA POWER (MERRA-2) data into a guided investigation. A student picks a place and a variable and gets three plain answers, each backed by a tested statistic: Is it changing (autocorrelation-corrected Mann-Kendall trend test)? Is it changing the same way everywhere (per-square trends with false discovery rate control, and POD pattern maps)? Are the worst days getting worse (extreme value analysis with bootstrap ranges)? Two demo cases, Florida (humid heat) and Phoenix region, Arizona (dry heat): same country, same NASA data, opposite kinds of heat. Florida's heat index rises more than twice as fast as its air temperature, and Phoenix is drying while it warms.

## How it addresses the challenge

- Variables worth checking, with plain definitions: 4 NASA POWER variables plus heat index.
- Regional data access: tiled, rate-limited, cached downloads from the POWER regional API.
- Long time series with the seasonal cycle removed (monthly anomalies, 1991-2020 normal).
- Rate of change with significance: slope per decade with a 95% range, corrected Mann-Kendall test, Sen's slope.
- Guided questions and demos: five-step investigation, margin notes for high school readers, "Watch a demo".
- Same process, different trends: POD patterns and per-square maps; Florida vs Phoenix humidity and rain trends go different ways.

## NASA data used

- NASA POWER daily regional API, community AG, variables T2M, T2M_MAX, RH2M, PRECTOTCORR, 1981-2025: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- MERRA-2 (source of the POWER meteorology).

## Other resources

Natural Earth (land, state and country shapes), ColorBrewer palettes, IBM Plex and Source Serif 4 fonts (OFL), Python (xarray, scipy, pymannkendall, regionmask, Streamlit, Altair).

## Use of AI

Code written with Claude Code (Anthropic) under the author's direction. All statistical choices were checked by simulation and unit tests (see README, Validation).

## 30-second demo video plan

1. (0-5 s) Open the live link, title and warming stripes.
2. (5-15 s) Click "Watch a demo": Florida heat index, step 3 verdict box (+0.86 °C per decade, SIGNIFICANT).
3. (15-23 s) Step 4: every square warming, Pattern 2 north vs south.
4. (23-30 s) Switch to Phoenix, step 5: the old 20-year heat level now about every 2 years, with the honest caution note.
