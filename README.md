# Mizan: is the trend real?

**NASA Space Apps Challenge 2026** · Challenge: *Be An Earth System Trend Detective!* · Universal Event

**Live app:** https://mizan-trends.streamlit.app

Mizan (ميزان, "scale") weighs the evidence before calling a change in NASA data a trend.
Pick a place and a variable. Mizan shows how it changed since 1981, how fast it is changing
per decade, and a verdict: **Robust**, **Inconclusive** or **Abrupt shift**, with the reasons.

## How it answers the challenge

| The challenge asks | Mizan's answer |
|---|---|
| What is changing? | Seven NASA POWER variables, by season |
| Where? | Any point on Earth, with 15 Saudi cities built in |
| How much? | Theil-Sen slope per decade with a 95% range |
| Is it statistically significant? | Mann-Kendall test corrected for autocorrelation, a stress test over every start and end year, a shift check, and false-discovery control across cities |

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest`  ·  Calibration simulation: `python calibration.py`  ·  Save all city data into `data/`: `python prefetch.py`

## Files

- `app.py` the web app (Streamlit + Plotly)
- `stats.py` the trend tests and the verdict logic
- `power.py` downloads and prepares NASA POWER data
- `prefetch.py` saves NASA POWER data for every built-in city into `data/`, so the app works if the API is down
- `scan.py` runs every city, variable and season and writes `results/all_results.csv`
- `calibration.py` measures how often each verdict is right on simulated data
- `tests/` automated checks

## NASA data

- **NASA POWER** monthly API, https://power.larc.nasa.gov/ (meteorology from NASA's MERRA-2 model, 1981 onward). Accessed 14-15 November 2026.

## Limits

- One POWER grid cell is about 50 km wide.
- Mizan misses weak trends and cannot always tell a single step from a gradual change (see the "How Mizan decides" tab).
- Each value is one grid cell about 50 km wide, so "Jeddah" means the Jeddah area.
- Rainfall verdicts in deserts are unreliable.
- A robust trend says that something changed, not why.

## Use of AI

Parts of this project were made with AI assistance (Claude by Anthropic): the first drafts of
`app.py`, `stats.py`, `power.py`, `prefetch.py`, `scan.py`, `calibration.py` and the tests, and editing help with text.
The team ran, tested and reviewed the code, chose the case studies and interpreted the results.
The team emblem is an AI-assisted drawing.

## Open-source libraries

Streamlit, Plotly, pandas, NumPy, SciPy, Requests, pytest.

## Team

ADD NAMES OR NICKNAMES AND ROLES
