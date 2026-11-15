"""
Calibration: how often does each verdict come out when we know the truth?

We simulate 45-year records (1981-2025, the length of the NASA POWER weather
record) where we decide the truth: pure noise, a gradual trend, or one step.
Each record goes through two tests:

  - Plain line: an ordinary least-squares (OLS) straight-line fit, called
    significant when its p-value is below 0.05. This is what a spreadsheet
    trend line reports.
  - Mizan: the full verdict from stats.judge (Robust, Inconclusive or Abrupt shift).

On noise, "significant" or "Robust" is a false alarm. On a trend, it is a detection.
On a step, "Robust" means Mizan read the step as a gradual trend.

Settings (fixed, so the table is reproducible):
  - 300 records per case, random seed 7, reset at the start of every case
  - noise has standard deviation 1 ("sd"), so trend and step sizes are in sd
  - trends have year-to-year autocorrelation (AR(1)) 0.3, as climate data usually does
  - steps happen in 2001, on independent noise

Run with:  python calibration.py          (about 5 minutes)
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


def ar1(rng, phi: float, sd: float = 1.0) -> np.ndarray:
    """Noise in which each year resembles the year before it with correlation phi."""
    e = rng.normal(0, sd, N)
    x = np.empty(N)
    x[0] = e[0] / np.sqrt(1 - phi ** 2) if phi < 1 else e[0]   # start in the steady state
    for i in range(1, N):
        x[i] = phi * x[i - 1] + e[i]
    return x


def trend(per_decade: float):
    return lambda rng: (per_decade / 10) * (YEARS - YEARS[0]) + ar1(rng, 0.3)


def step(size: float):
    return lambda rng: ar1(rng, 0.0) + size * (YEARS >= 2001)


# Same order as the README table and slide 6.
CASES = [
    ("Pure noise, independent years",       lambda rng: ar1(rng, 0.0)),
    ("Pure noise, autocorrelation 0.3",     lambda rng: ar1(rng, 0.3)),
    ("Pure noise, autocorrelation 0.5",     lambda rng: ar1(rng, 0.5)),
    ("Weak trend (0.3 sd per decade)",      trend(0.3)),
    ("Moderate trend (0.6 sd per decade)",  trend(0.6)),
    ("Clear trend (1.0 sd per decade)",     trend(1.0)),
    ("Small single step (1 sd)",            step(1.0)),
    ("Large single step (2 sd)",            step(2.0)),
]


def run(generate) -> dict:
    rng = np.random.default_rng(SEED)
    plain = robust = shift = 0
    for _ in range(REPS):
        x = generate(rng)
        plain += stats.naive_p(YEARS, x) < stats.ALPHA
        label = stats.judge(YEARS, x).label
        robust += label == stats.ROBUST
        shift += label == stats.SHIFT
    return {"plain": plain / REPS, "robust": robust / REPS, "shift": shift / REPS}


def pct(share: float) -> str:
    """Whole percent, with one decimal below 1% so a rare outcome does not show as 0%."""
    value = share * 100
    if value == 0:
        return "0%"
    return f"{value:.1f}%" if value < 1 else f"{value:.0f}%"


def main() -> None:
    rows = []
    for name, generate in CASES:
        result = run(generate)
        rows.append((name, result))
        print(f"  done: {name}", file=sys.stderr)

    header = "| Simulated truth | Plain line significant | Mizan: Robust | Mizan: Abrupt shift |"
    print(header)
    print("|---|---|---|---|")
    for name, r in rows:
        print(f"| {name} | {pct(r['plain'])} | {pct(r['robust'])} | {pct(r['shift'])} |")

    if "--csv" in sys.argv:
        Path("results").mkdir(exist_ok=True)
        with open("results/calibration.csv", "w") as out:
            out.write("case,plain_line_significant,mizan_robust,mizan_abrupt_shift,records\n")
            for name, r in rows:
                out.write(f"\"{name}\",{r['plain']:.4f},{r['robust']:.4f},{r['shift']:.4f},{REPS}\n")
        print("\nWrote results/calibration.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
