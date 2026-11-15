# Mizan: is the trend real?

**NASA Space Apps Challenge 2026** · Challenge: *Be An Earth System Trend Detective!* · Universal Event

**Live app:** https://mizan-trends.streamlit.app

**Slides (7-slide demo, PDF):** [docs/Mizan-slides.pdf](docs/Mizan-slides.pdf)
(made before the November 2026 review round; where the slides and this README differ, this README is current)

Mizan weighs the evidence before calling a change in NASA data a trend.
Pick a place and a variable. Mizan shows how it has changed since 1981, how fast it is changing
per decade, and a verdict with its reasons: **Robust**, **Inconclusive** or **Abrupt shift**.
It is a conservative screen: it raises far fewer false alarms than a spreadsheet trend line, and pays for
that by missing many weak trends. New users can start from three worked examples in the sidebar.

![Mizan comparing air temperature trends across 15 areas of Saudi Arabia](docs/compare-cities.png)

## Why

A rising line is easy to draw and easy to misread. A spreadsheet trend line assumes every year is
independent and accepts whatever start year you give it. Mizan is for people who have to judge a
claim about a changing climate without a statistics degree: journalists, teachers, students and planners.

## How it answers the challenge

| The challenge asks | Mizan's answer |
|---|---|
| What is changing? | Seven NASA POWER variables, by season |
| Where? | Any point on Earth, with 15 areas of Saudi Arabia built in and mapped |
| How much? | Theil-Sen slope per decade with a 95% range |
| Is it statistically significant? | The checks listed below, plus false-discovery control across places, all calibrated on simulated records |
| The same variable moving in opposite directions | An "opposite twins" search across places |

## The decision rule

The same checks, in the same order and with the same thresholds, for every place, variable and season.

| Check | Question | Passes when |
|---|---|---|
| Record length | Is there enough data? | At least 20 complete years |
| Shift check | Is it one jump? | A single step does not fit clearly better than a line (BIC difference of 6 or less, counted on the effectively independent years), or a significant same-direction trend exists on one side of the step |
| Noise test | Could random variation explain it? | Mann-Kendall p below 0.05 after a lag-1 autocorrelation correction, and the 95% range of the slope excludes zero. If the shift check found a jump, the noise is measured around the jump, not around a line |
| Stress test | Does it survive other start and end years? | At least 90% of all periods of 20+ years have the same direction as the full record (records under 30 years, possible only for a custom point: periods of at least 60% of the record) |
| Source check (sunlight only) | Is it just the 2001 switch of data product? | A trend fits better than two flat levels split at 2001, or the trend continues on one side of 2001 |

- **Robust:** passes every check. It holds under these rules; it is not proof.
- **Inconclusive:** fails the noise test, the stress test or the source check, or (when many places are tested together) the many-places correction. This does not mean nothing is changing. It means this record cannot show it.
- **Abrupt shift:** significant, but concentrated in one jump. It may be a real shift or a change in the data record.

Three design choices, and why:

- **The shift check runs first.** A jump makes the leftovers around a straight line look like slow, persistent
  noise. Measured that way, the autocorrelation correction swells (up to 7× for Tabuk sunlight) and a clear jump
  is reported as "random variation" for the wrong reason. When the shift check finds a jump, the noise is measured
  around the jump instead. The step comparison itself uses the effective number of independent years, because
  autocorrelated noise wanders and can mimic a jump.
- **The 90% rule is a stability screen, not a significance test.** The periods overlap heavily, so 90% is a tuning
  constant, not a confidence level; its effect is measured only by the calibration below. It guards against
  cherry-picked start years, and it also marks change that only starts late in the record as unstable.
- **The sunlight switch is tested directly.** The shift check reports only the single best jump anywhere in the
  record, so a smaller jump at the 2001 GEWEX SRB to CERES switch can hide inside a record and pass as a trend.

## What we found

We ran Mizan on 15 areas × 7 variables × 5 seasons = 525 tests (`scan.py`, results in `results/all_results.csv`).
Each variable and season is one family of 15 areas for the Benjamini-Hochberg correction (35 families; no
correction across all 525).

| Finding | Detail |
|---|---|
| Mizan and a plain trend line disagree often | An ordinary straight-line fit calls 319 of the 525 tests significant. Mizan agrees on 207 (190 Robust, 17 Abrupt shift) and disagrees on 112: 47 fail the noise test, 30 fail the stress test, 34 do not survive the many-places correction, and 1 is explained by the 2001 sunlight switch. Mizan never calls significant what the plain line does not. The truth is unknown on real data, so some of the 112 may be real trends Mizan cannot confirm. |
| Warming is everywhere | Yearly air temperature rises in all 15 areas, by 0.24 to 0.49 °C per decade. Fourteen are Robust and one (Abha) is an Abrupt shift around 1998. |
| The sunlight record is not one record | For whole-year sunlight, two flat levels split at the 2001 product switch fit as well as a trend, with no trend on either side, in 10 of the 15 areas. No whole-year sunlight trend is Robust. The clearest trap: winter sunlight in Dammam, where a plain line reports p = 0.003 and Mizan's earlier version said Robust; the source check shows the switch explains it. |
| Shifts cluster in time | Of 17 abrupt shifts, 11 fall in 1998-1999 (7 temperature in 4 areas, 4 sunlight in 3 distinct records), 4 in 1994 (winter and spring sunlight in Abha, Hail, Buraydah and Tabuk) and 2 in 1987-1989 (wind). |
| More jumps sit in records that fail the noise test | The shift check also fires in 73 Inconclusive records. Their years cluster too: sunlight in 1997-1998 (15 of 27), wind in 1988-1989 (6 of 8), rainfall in 1987-1988 (5 of 32) and at the very end of the record, 2020-2021 (23 of 32). |
| Rainfall and humidity | No robust trend in any area or season (0 of 150 tests). |
| No opposite twins | No variable and season has two Saudi areas with robust trends in opposite directions. |

**How to read the shift clusters.** The counts overstate the evidence. Seasons are nested inside the whole year,
`T2M` and `T2M_MAX` move together, neighbouring areas share weather (Dammam and Al-Ahsa), and some share a grid
cell (Jeddah and Makkah sunlight). The 11 shifts in 1998-1999 are about 7 independent records, not 11.

A year shared by different places points either to a real regional change or to a change in the record behind
the data. We have not established which. The candidates we would test, data-record explanations first:

- **Sunlight, 1994 and 1998.** NASA POWER sunlight before 2001 comes from GEWEX SRB, which takes its clouds from
  ISCCP. ISCCP cloud records have documented spurious changes tied to satellite coverage and viewing angle
  (Evan, Heidinger & Vimont, 2007). The geostationary view over Arabia changed in 1998, when Meteosat-5 began
  covering the Indian Ocean from 63°E; we have not confirmed that SRB used those images. The 1991 Pinatubo
  eruption is a weak candidate: in Tabuk the lowest sunlight years are 1984-1988, before the eruption.
- **Temperature, 1998-1999.** MERRA-2 began assimilating ATOVS (AMSU-A) satellite data from NOAA-15 in late 1998.
  Physical candidates are the strong 1997-98 El Niño and the La Niña and dry years that followed. The decisive
  test, which we have not done, is to compare with independent station records for two or three of these areas.
- **Wind and rainfall, 1987-1989.** The SSM/I microwave instrument entered the reanalysis in 1987; NASA GMAO
  reports that MERRA-2 rainfall still shows a jump then.
- **Rainfall, 2020-2021.** A break this close to the end of the record rests on only a few years and should not be read.

## How well calibrated is Mizan?

`calibration.py` runs Mizan, and an ordinary straight-line fit, on simulated 45-year records where the truth
is known (300 records per case, noise with standard deviation 1). "Any significant change" (Robust or Abrupt
shift) is the like-for-like comparison with the plain line, which has no shift category. With 300 records a
rate of 6% is uncertain by about ±1.4 points.

| Simulated truth | Plain line significant | Mizan: Robust | Mizan: Abrupt shift | Mizan: any significant change |
|---|---|---|---|---|
| Pure noise, independent years | 7% | 3% | 0% | 3% |
| Pure noise, autocorrelation 0.3 | 17% | 3% | 0.3% | 3% |
| Pure noise, autocorrelation 0.5 | 31% | 4% | 0.7% | 5% |
| Pure noise, autocorrelation 0.7 | 44% | 6% | 0.7% | 6% |
| Weak trend (0.3 sd per decade), autocorr. 0.3 | 70% | 34% | 2% | 37% |
| Moderate trend (0.6 sd per decade), autocorr. 0.3 | 100% | 82% | 4% | 85% |
| Moderate trend (0.6 sd per decade), autocorr. 0.5 | 99% | 67% | 4% | 71% |
| Clear trend (1.0 sd per decade), autocorr. 0.3 | 100% | 98% | 1% | 99% |
| Small step (1 sd) in 2001, independent | 81% | 56% | 6% | 63% |
| Small step (1 sd) in 2001, autocorr. 0.3 | 77% | 38% | 6% | 45% |
| Small step (1 sd) in 1990, independent | 45% | 17% | 3% | 20% |
| Large step (2 sd) in 2001, independent | 100% | 44% | 54% | 98% |
| Large step (2 sd) in 2001, autocorr. 0.3 | 100% | 52% | 31% | 84% |
| Moderate trend + 0.5 sd step in 2001, autocorr. 0.3 | 100% | 91% | 7% | 98% |
| Source change only: 0.75 sd step in 2001, autocorr. 0.3, no break check | 56% | 26% | 3% | 29% |
| Source change only: 0.75 sd step in 2001, autocorr. 0.3, with 2001 break check | 56% | 12% | 3% | 14% |
| Moderate trend, autocorr. 0.3, with 2001 break check | 100% | 73% | 4% | 76% |

On pure noise Mizan's false-alarm rate stays at about the nominal 5%, even when neighbouring years strongly
resemble each other; a plain line raises a false alarm up to 44% of the time. The costs: Mizan confirms only about
a third of weak trends, and it often reads a step as a gradual trend (a 1 sd step is called Robust 38-56% of the
time, a 2 sd step about half the time). The 2001 source check halves the share of pure source-change jumps that
pass as Robust (26% to 12%), at a cost of about 9 points of detection on a real trend (82% to 73%).

Compared with the slides, this table adds autocorrelated steps, steps off the middle of the record, a trend plus a
step, stronger autocorrelation and the source-change cases, and it scales noise so its standard deviation is 1
whatever the autocorrelation. The numbers therefore differ slightly from the slides.

## Limits

- Each value is one NASA POWER grid cell: 0.5° × 0.625° (about 55 km) for weather from MERRA-2, and 1° × 1° (about 110 km) for sunlight, so Jeddah and Makkah share one sunlight record. "Jeddah" means the Jeddah area. Near a coast the cell mixes land and sea, and it cannot see a city's own heat island.
- Rainfall verdicts in deserts are unreliable: mostly zeros, rare storms, and weak model rainfall.
- Records start in 1981 (1984 for sunlight).
- The sunlight record joins two products: GEWEX SRB up to 2000 and CERES from 2001. NASA advises against
  using it to assess climate trends across a change of source. Mizan tests the switch year directly, but a
  smaller jump at the switch can still pass unnoticed (see the calibration table).
- The weather comes from a model (MERRA-2) whose satellite inputs changed over time. A jump can come from the record, not the climate.
- Verdicts near a threshold depend on the exact years. Starting the record 2 years later, or ending it 2 years earlier, changes 49 of the 525 verdicts (9%), including 23 of the 190 Robust ones (`python scan.py --stability`).
- A robust trend says that something changed, not why.
- The "Any point on Earth" option needs a live connection to NASA POWER. The 15 built-in areas use saved copies in `data/`.
- The many-places correction covers one variable and season at a time, not all 525 tests together.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

| Command | What it does |
|---|---|
| `python -m pytest` | Runs the automated checks |
| `python scan.py` | Re-runs all 525 tests from the saved data (about 5 minutes; add `--stability` to also re-run them with 2 years trimmed from either end) |
| `python calibration.py` | Re-runs the calibration simulation and prints the table above (about 15 minutes; add `--csv` to save `results/calibration.csv`, which the app shows) |
| `python prefetch.py` | Downloads the built-in areas from NASA POWER into `data/` |

## Files

- `app.py` the web app (Streamlit + Plotly)
- `stats.py` the trend tests and the verdict logic
- `power.py` downloads and prepares NASA POWER data
- `scan.py`, `calibration.py`, `prefetch.py` the scripts in the table above
- `data/` saved NASA POWER records for the built-in areas, each with its request URL and access date
- `results/all_results.csv` every verdict from the scan
- `results/calibration.csv` the calibration table above
- `docs/Mizan-slides.pdf` the 7-slide project demo
- `tests/` automated checks
- `requirements.txt` exact versions the results were produced with

## NASA data

**NASA POWER** (Prediction Of Worldwide Energy Resources), NASA Langley Research Center, monthly API:
https://power.larc.nasa.gov/ . Meteorological values come from NASA's MERRA-2 model. Sunlight
(`ALLSKY_SFC_SW_DWN`) comes from NASA GEWEX SRB Release 4.0-IP (1984-2000), then CERES SYN1deg
(2001 onwards), as listed in POWER's data sources:
https://power.larc.nasa.gov/docs/methodology/data/sources_fluxes/ .
Accessed 14 November 2026. Parameters: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `WS10M`, `ALLSKY_SFC_SW_DWN`.

## Methods we relied on

- Sen (1968) and Theil (1950): the median-of-slopes trend estimate.
- Mann (1945) and Kendall (1975): the rank-based trend test.
- Yue & Wang (2004): the autocorrelation correction. Mizan uses its large-sample lag-1 (AR(1)) form, (1 + ρ) / (1 − ρ). It does not implement the multi-lag modified test of Hamed & Rao (1998).
- Schwarz (1978): the Bayesian information criterion used to compare a step with a line.
- Benjamini & Hochberg (1995): false discovery rate control across many places.
- Evan, Heidinger & Vimont (2007), Geophysical Research Letters: spurious changes in ISCCP cloud records.

## Use of AI

We used Claude (Anthropic) as a coding and writing assistant. Files it helped draft say so in their header.
The team tested the code, chose what to investigate, and checked the results.

## Open-source libraries

Streamlit, Plotly, pandas, NumPy, SciPy, Requests, pytest.

## Team

Team Mizan, NASA Space Apps Challenge 2026, Universal Event.

## License

MIT. See `LICENSE`.
