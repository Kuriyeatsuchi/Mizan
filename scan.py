"""
Run Mizan on every built-in area, variable and season and save the results.
Uses the saved copies in data/, so it needs no internet.

Run with:  python scan.py               ->  results/all_results.csv
           python scan.py --stability   also re-judges everything with the first 2 or the
                                        last 2 years removed and reports how many verdicts
                                        change (about 15 minutes)

Each variable and season is one family of 15 areas. Within a family, p-values
are adjusted with Benjamini-Hochberg (identical records, such as Jeddah and
Makkah sunlight, counted once), and a Robust or Abrupt shift verdict that does
not survive the adjustment becomes Inconclusive with the note "many_places".
The app's "Compare Saudi areas" tab uses the same function, judge_places().

Columns of results/all_results.csv:
  verdict, note        final verdict and its main reason
  slope10, lo, hi      Theil-Sen change per decade and its 95% range
  p, adj_p             Mizan's noise-test p-value, before and after Benjamini-Hochberg
  plain_p              p-value of an ordinary straight-line fit
  agree                stress-test agreement (share of 20+ year periods with the same sign)
  step_year            year of a strong single jump if the step check found one,
                       whatever the final verdict
  source_change        True when the 2001 sunlight product switch explains the change
                       (empty for variables without a switch)

AI assistance: drafted with Claude (Anthropic) during the hackathon. See README.md.
"""
from pathlib import Path

import pandas as pd

import sys

import power
import stats


def judge_places(parameter, season, load=power.fetch, progress=None):
    """
    Judge every built-in area for one variable and season, then adjust across them.
    Returns (rows, errors): one dict per area, and a list of "area: problem" texts.
    """
    months = power.SEASONS[season]
    known_break = power.SOURCE_CHANGE.get(parameter)
    rows, verdicts, series, errors = [], [], [], []
    for i, (city, (lat, lon)) in enumerate(power.CITIES.items()):
        try:
            table, meta = load(lat, lon, parameter)
            y, x = power.yearly(table, months)
            v = stats.judge(y, x, known_break=known_break)
        except Exception as err:   # one area failing should not stop the others
            errors.append(f"{city}: {err}")
            continue
        finally:
            if progress:
                progress((i + 1) / len(power.CITIES))
        if v.trend is None:
            continue
        rows.append(dict(param=parameter, season=season, city=city, lat=lat, lon=lon,
                         slope10=v.trend.slope * 10, lo=v.trend.slope_low * 10, hi=v.trend.slope_high * 10,
                         p=v.trend.p, plain_p=stats.naive_p(y, x), agree=v.windows.agreement,
                         step_year=v.step.break_year if v.step.is_step else None,
                         source_change=(v.source.explains if v.source else None),
                         units=meta["units"]))
        verdicts.append(v)
        series.append((y, x))
    adjusted, keep = stats.adjust_places([v.trend.p for v in verdicts], series)
    stats.apply_places(verdicts, keep)
    for row, v, a in zip(rows, verdicts, adjusted):
        row.update(verdict=v.label, note=v.notes[0] if v.notes else "", adj_p=a)
    return rows, errors


COLUMNS = ["param", "season", "city", "verdict", "note", "slope10", "lo", "hi", "p", "adj_p",
           "plain_p", "agree", "step_year", "source_change", "units"]


def trimmed(first: int = 0, last: int = 0):
    """A loader that drops whole years from the start or end of each saved record."""
    def load(lat, lon, parameter):
        table, meta = power.fetch(lat, lon, parameter)
        lo, hi = table["year"].min() + first, table["year"].max() - last
        return table[(table["year"] >= lo) & (table["year"] <= hi)], meta
    return load


def stability(full: pd.DataFrame) -> None:
    """How many verdicts change if the record starts 2 years later or ends 2 years earlier."""
    key = ["param", "season", "city"]
    changed = pd.Series(False, index=pd.MultiIndex.from_frame(full[key]))
    for name, load in (("first 2 years removed", trimmed(first=2)), ("last 2 years removed", trimmed(last=2))):
        rows = []
        for param in power.VARIABLES:
            for season in power.SEASONS:
                rows += judge_places(param, season, load=load)[0]
        other = pd.DataFrame(rows).set_index(key)["verdict"].reindex(changed.index)
        diff = other.to_numpy() != full.set_index(key)["verdict"].reindex(changed.index).to_numpy()
        print(f"{name}: {diff.sum()} of {len(diff)} verdicts change")
        changed |= diff
    robust = (full["verdict"] == stats.ROBUST).to_numpy()
    print(f"either: {changed.sum()} of {len(changed)} verdicts change; "
          f"{(changed.to_numpy() & robust).sum()} of {robust.sum()} Robust verdicts change")


def main():
    rows = []
    for param in power.VARIABLES:
        for season in power.SEASONS:
            block, errors = judge_places(param, season)
            for e in errors:
                print("skipped", param, season, e)
            rows += block
    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)[COLUMNS]
    df.to_csv("results/all_results.csv", index=False)
    print(len(df), "results")
    print(df.groupby(["param", "verdict"]).size().unstack(fill_value=0))
    if "--stability" in sys.argv:
        stability(df)


if __name__ == "__main__":
    main()
