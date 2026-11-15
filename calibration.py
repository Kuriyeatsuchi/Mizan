"""
Calibration: how often does each verdict come out when we know the truth?

We simulate 45-year records (1981-2025, the length of the NASA POWER weather
record) where we decide the truth: pure noise, a gradual trend, one step, or a
mix. Each record goes through two tests:

  - Plain line: an ordinary least-squares (OLS) straight-line fit, called
    significant when its p-value is below 0.05. This is what a spreadsheet
    trend line reports.
  - Mizan: the full verdict from stats.judge (Robust, Inconclusive or Abrupt shift).

Columns:
  - "Plain line significant": the OLS p-value is below 0.05.
  - "Mizan: Robust": a gradual trend.
  - "Mizan: Abrupt shift": a significant change concentrated in one step.
  - "Mizan: any significant change": Robust or Abrupt shift. This is the
    like-for-like comparison with the plain line, which has no shift category.

On noise, a significant result is a false alarm. On a trend, it is a detection.
On a step, "Robust" means Mizan read the step as a gradual trend.

Settings (fixed, so the table is reproducible):
  - 300 records per case, random seed 7, reset at the start of every case.
    With 300 records a rate of 6% has a standard error of about 1.4 points,
    so 3%, 5% and 6% are not clearly different.
  - noise has standard deviation 1 ("sd") whatever its autocorrelation, so
    trend and step sizes are in sd of the year-to-year values
  - "source-change" cases pass known_break=2001 to stats.judge, as the app does
    for NASA POWER sunlight (GEWEX SRB up to 2000, CERES from 2001)

Run with:  python calibration.py          (about 10 minutes)
           python calibration.py --csv    (also writes results/calibration.csv)

AI assistance: drafted with Claude (Anthropic) during the hackathon and reviewed
and tested by the team. See README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import stats

YEARS = np.arange(1981, 2026, dtype=float)   # 45 years
N = len(YEARS)
REPS = 300
SEED = 7


def ar1(rng, phi: float) -> np.ndarray:
    """
    Noise in which each year resembles the year before it with correlation phi.
    Scaled so the year-to-year values have standard deviation 1 for any phi.
    """
    e = rng.normal(0, np.sqrt(1 - phi ** 2), N)
    x = np.empty(N)
    x[0] = rng.normal(0, 1)            # start in the steady state
    for i in range(1, N):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make(per_decade: float = 0.0, step: float = 0.0, step_year: int = 2001, phi: float = 0.0):
    """A gradual trend plus one step plus AR(1) noise."""
    def generate(rng):
        return ((per_decade / 10) * (YEARS - YEARS[0]) + step * (YEARS >= step_year) + ar1(rng, phi))
    return generate


# (name, generator, known_break). Same order as the README table.
CASES = [
    ("Pure noise, independent years",                       make(), None),
    ("Pure noise, autocorrelation 0.3",                     make(phi=0.3), None),
    ("Pure noise, autocorrelation 0.5",                     make(phi=0.5), None),
    ("Pure noise, autocorrelation 0.7",                     make(phi=0.7), None),
    ("Weak trend (0.3 sd per decade), autocorr. 0.3",       make(0.3, phi=0.3), None),
    ("Moderate trend (0.6 sd per decade), autocorr. 0.3",   make(0.6, phi=0.3), None),
    ("Moderate trend (0.6 sd per decade), autocorr. 0.5",   make(0.6, phi=0.5), None),
    ("Clear trend (1.0 sd per decade), autocorr. 0.3",      make(1.0, phi=0.3), None),
    ("Small step (1 sd) in 2001, independent",              make(step=1.0), None),
    ("Small step (1 sd) in 2001, autocorr. 0.3",            make(step=1.0, phi=0.3), None),
    ("Small step (1 sd) in 1990, independent",              make(step=1.0, step_year=1990), None),
    ("Large step (2 sd) in 2001, independent",              make(step=2.0), None),
    ("Large step (2 sd) in 2001, autocorr. 0.3",            make(step=2.0, phi=0.3), None),
    ("Moderate trend + 0.5 sd step in 2001, autocorr. 0.3", make(0.6, step=0.5, phi=0.3), None),
    ("Source change only: 0.75 sd step in 2001, autocorr. 0.3, no break check", make(step=0.75, phi=0.3), None),
    ("Source change only: 0.75 sd step in 2001, autocorr. 0.3, with 2001 break check", make(step=0.75, phi=0.3), 2001),
    ("Moderate trend, autocorr. 0.3, with 2001 break check", make(0.6, phi=0.3), 2001),
]


def run(generate, known_break=None) -> dict:
    rng = np.random.default_rng(SEED)
    plain = robust = shift = 0
    for _ in range(REPS):
        x = generate(rng)
        plain += stats.naive_p(YEARS, x) < stats.ALPHA
        label = stats.judge(YEARS, x, known_break=known_break).label
        robust += label == stats.ROBUST
        shift += label == stats.SHIFT
    return {"plain": plain / REPS, "robust": robust / REPS, "shift": shift / REPS,
            "any": (robust + shift) / REPS}


def pct(share: float) -> str:
    """Whole percent, with one decimal below 1% so a rare outcome does not show as 0%."""
    value = share * 100
    if value == 0:
        return "0%"
    return f"{value:.1f}%" if value < 1 else f"{value:.0f}%"


def main() -> None:
    rows = []
    only = [a for a in sys.argv[1:] if not a.startswith("--")]   # optional: run cases whose name contains this text
    for name, generate, known_break in CASES:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        result = run(generate, known_break)
        rows.append((name, result))
        print(f"  done: {name}", file=sys.stderr)

    print("| Simulated truth | Plain line significant | Mizan: Robust | Mizan: Abrupt shift | Mizan: any significant change |")
    print("|---|---|---|---|---|")
    for name, r in rows:
        print(f"| {name} | {pct(r['plain'])} | {pct(r['robust'])} | {pct(r['shift'])} | {pct(r['any'])} |")

    if "--csv" in sys.argv:
        Path("results").mkdir(exist_ok=True)
        with open("results/calibration.csv", "w") as out:
            out.write("case,plain_line_significant,mizan_robust,mizan_abrupt_shift,mizan_any_significant,records\n")
            for name, r in rows:
                out.write(f"\"{name}\",{r['plain']:.4f},{r['robust']:.4f},{r['shift']:.4f},{r['any']:.4f},{REPS}\n")
        print("\nWrote results/calibration.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
