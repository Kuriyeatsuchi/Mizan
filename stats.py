"""
Mizan statistics: is a change in a yearly series a real trend?

Everything here works on two plain arrays:
    t = the years (e.g. 1981, 1982, ...)
    x = one value per year (e.g. mean summer temperature)

The checks, in the order the app uses them:
    1. Theil-Sen slope      -> how big is the change (robust to outliers)
    2. Mann-Kendall test    -> could noise alone explain it (corrected for autocorrelation)
    3. Step check           -> is it one sudden jump instead of a gradual trend
    4. Window stress test   -> does the direction survive other start and end years
    5. Benjamini-Hochberg   -> fair significance when many places are tested at once

AI assistance: this file was drafted with Claude (Anthropic) during the hackathon
and reviewed and tested by the team. See README.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import linregress, norm

ALPHA = 0.05            # significance level used everywhere
MIN_YEARS = 20          # shorter records get "Inconclusive" without testing
AGREEMENT_NEEDED = 0.90 # share of sub-periods that must agree on the direction
STEP_BIC_MARGIN = 6.0   # how much better a step must fit than a line ("strong" evidence)


# ---------------------------------------------------------------- trend size and test

@dataclass
class Trend:
    n: int
    slope: float          # units per year
    slope_low: float      # 95% range, units per year
    slope_high: float
    intercept: float
    s: float              # Mann-Kendall S statistic
    z: float
    p: float              # two-sided p-value after autocorrelation correction
    variance_factor: float  # 1.0 = no correction was needed

    @property
    def significant(self) -> bool:
        return self.p < ALPHA and (self.slope_low > 0 or self.slope_high < 0)


def _mk_s(x: np.ndarray) -> float:
    """Mann-Kendall S: (pairs that go up) minus (pairs that go down)."""
    signs = np.sign(x[None, :] - x[:, None])   # signs[i, j] = sign(x[j] - x[i])
    return float(np.triu(signs, k=1).sum())


def _mk_variance(x: np.ndarray) -> float:
    """Variance of S if the values were independent, with the correction for ties."""
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    ties = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    return (n * (n - 1) * (2 * n + 5) - ties) / 18.0


def _pairwise_slopes(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(x), k=1)
    return (x[j] - x[i]) / (t[j] - t[i])


def _autocorrelation_factor(t: np.ndarray, x: np.ndarray, slope: float) -> float:
    """
    Variance correction for autocorrelation (lag-1 effective sample size).

    Years that resemble their neighbours carry less independent information than
    the plain test assumes. We remove the trend, measure how strongly each year
    resembles the year before it (rho), and inflate the variance of S by
    (1 + rho) / (1 - rho). The same idea is behind the corrections of
    Hamed & Rao (1998) and Yue & Wang (2004).

    Two cautious choices: rho gets the usual small-sample bias correction, and
    the variance is never shrunk (factor >= 1).
    """
    n = len(x)
    r = x - slope * (t - t.mean())
    r = r - r.mean()
    denom = float((r ** 2).sum())
    if denom == 0 or n < 5:
        return 1.0
    rho = float((r[:-1] * r[1:]).sum()) / denom
    rho = rho + (1 + 4 * rho) / n          # lag-1 estimates run low in short series
    rho = min(rho, 0.9)
    if rho <= 0:
        return 1.0
    return (1 + rho) / (1 - rho)


def trend_test(t, x) -> Trend:
    """Theil-Sen slope with a 95% range, and the corrected Mann-Kendall test."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 years to test a trend.")

    slopes = np.sort(_pairwise_slopes(t, x))
    slope = float(np.median(slopes))
    intercept = float(np.median(x) - slope * np.median(t))

    s = _mk_s(x)
    factor = _autocorrelation_factor(t, x, slope)
    variance = _mk_variance(x) * factor

    if variance <= 0 or s == 0:
        z = 0.0
    else:
        z = (s - np.sign(s)) / math.sqrt(variance)   # continuity correction
    p = float(2 * (1 - norm.cdf(abs(z))))

    # 95% range for the slope, using the same corrected variance (Sen, 1968)
    count = len(slopes)
    c = norm.ppf(1 - ALPHA / 2) * math.sqrt(max(variance, 0.0))
    low_i = int(round((count - c) / 2)) - 1
    high_i = int(round((count + c) / 2))
    low = float(slopes[min(max(low_i, 0), count - 1)])
    high = float(slopes[min(max(high_i, 0), count - 1)])

    return Trend(n, slope, low, high, intercept, s, float(z), p, factor)


def naive_p(t, x) -> float:
    """p-value of an ordinary straight-line fit: what a spreadsheet trend line reports."""
    return float(linregress(np.asarray(t, dtype=float), np.asarray(x, dtype=float)).pvalue)


# ---------------------------------------------------------------- step check

@dataclass
class Step:
    is_step: bool
    break_year: float | None   # first year of the "after" part
    jump: float                # mean after minus mean before
    bic_margin: float          # > 0 means a step fits better than a straight line


def step_check(t, x) -> Step:
    """
    Compare two explanations of the series:
        a straight line (gradual change), or
        one sudden jump between two flat levels.
    A jump can be a real shift in the climate, or a change in what the model was
    fed. Either way it is not a gradual trend, so the app reports it separately.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    edge = 5
    if n < 2 * edge + 2:
        return Step(False, None, 0.0, 0.0)

    line = np.polyfit(t, x, 1)
    rss_line = float(((x - np.polyval(line, t)) ** 2).sum()) + 1e-12

    best_rss, best_k = math.inf, None
    for k in range(edge, n - edge + 1):
        before, after = x[:k], x[k:]
        rss = float(((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum())
        if rss < best_rss:
            best_rss, best_k = rss, k
    best_rss += 1e-12

    bic_line = n * math.log(rss_line / n) + 2 * math.log(n)
    bic_step = n * math.log(best_rss / n) + 3 * math.log(n)   # 2 levels + the break position
    margin = bic_line - bic_step
    jump = float(x[best_k:].mean() - x[:best_k].mean())

    is_step = False
    if margin > STEP_BIC_MARGIN:
        # A real gradual trend should still show up inside at least one of the two parts.
        direction = np.sign(jump)
        trend_inside = False
        for part_t, part_x in ((t[:best_k], x[:best_k]), (t[best_k:], x[best_k:])):
            if len(part_x) >= 8:
                part = trend_test(part_t, part_x)
                if part.p < ALPHA and np.sign(part.slope) == direction:
                    trend_inside = True
        is_step = not trend_inside

    return Step(is_step, float(t[best_k]), jump, float(margin))


# ---------------------------------------------------------------- window stress test

@dataclass
class Windows:
    start: np.ndarray
    end: np.ndarray
    slope: np.ndarray
    significant: np.ndarray
    agreement: float      # share of long windows with the same direction as the full record
    long_length: int      # what "long" meant, in years


def window_test(t, x, shortest: int = 15) -> Windows:
    """Re-run the trend test for every start and end year at least `shortest` years apart."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    full_sign = np.sign(trend_test(t, x).slope)
    long_length = MIN_YEARS if n >= 30 else max(10, int(n * 0.6))
    shortest = min(shortest, long_length)

    starts, ends, slopes, sig = [], [], [], []
    for i in range(0, n - shortest + 1):
        for j in range(i + shortest - 1, n):
            result = trend_test(t[i:j + 1], x[i:j + 1])
            starts.append(t[i]); ends.append(t[j])
            slopes.append(result.slope); sig.append(result.significant)
    starts, ends = np.array(starts), np.array(ends)
    slopes, sig = np.array(slopes), np.array(sig)

    long = (ends - starts + 1) >= long_length
    agreement = float((np.sign(slopes[long]) == full_sign).mean()) if long.any() and full_sign != 0 else 0.0
    return Windows(starts, ends, slopes, sig, agreement, long_length)


# ---------------------------------------------------------------- verdict

ROBUST, INCONCLUSIVE, SHIFT = "Robust", "Inconclusive", "Abrupt shift"


@dataclass
class Verdict:
    label: str
    trend: Trend | None = None
    step: Step | None = None
    windows: Windows | None = None
    notes: list[str] = field(default_factory=list)   # short machine-readable reasons


def judge(t, x) -> Verdict:
    """Weigh the evidence and return one of: Robust, Inconclusive, Abrupt shift."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_YEARS:
        return Verdict(INCONCLUSIVE, notes=["too_short"])

    trend = trend_test(t, x)
    step = step_check(t, x)
    windows = window_test(t, x)
    notes = []

    if not trend.significant:
        return Verdict(INCONCLUSIVE, trend, step, windows, ["not_significant"])
    if step.is_step:
        return Verdict(SHIFT, trend, step, windows, ["step"])
    if windows.agreement < AGREEMENT_NEEDED:
        return Verdict(INCONCLUSIVE, trend, step, windows, ["unstable"])
    return Verdict(ROBUST, trend, step, windows, notes)


# ---------------------------------------------------------------- many places at once

def benjamini_hochberg(p_values, q: float = ALPHA):
    """
    False discovery rate control. Returns (adjusted p-values, keep flags).

    Testing 15 cities at the 5% level would produce a false alarm fairly often by
    chance alone. The adjusted p-value of each city accounts for how many were tested.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p, np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # keep them monotone
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted, adjusted < q
