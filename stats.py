"""
Mizan statistics: is a change in a yearly series a real trend?

Everything here works on two plain arrays:
    t = the years (e.g. 1981, 1982, ...)
    x = one value per year (e.g. mean summer temperature)

The checks, in the order the app uses them:
    1. Theil-Sen slope      -> how big is the change (robust to outliers)
    2. Step check           -> is it one sudden jump instead of a gradual trend
    3. Mann-Kendall test    -> could noise alone explain it (lag-1 AR(1) variance
                               inflation, the large-sample form of Yue & Wang 2004).
                               When the step check finds a strong jump, the noise is
                               measured around the jump, not around a straight line.
    4. Window stress test   -> does the direction survive other start and end years
                               (a stability screen, not a significance test)
    5. Source-change check  -> for records that switch data product at a known year
                               (sunlight, 2001), is the change just that switch
    6. Benjamini-Hochberg   -> fair significance when many places are tested at once
                               (used by the 15-area comparison and scan.py, never by
                               a single-place query)

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


def _autocorrelation_factor(t: np.ndarray, x: np.ndarray, slope: float,
                            break_year: float | None = None) -> float:
    """
    Lag-1 (AR(1)) variance inflation for autocorrelation.

    Years that resemble their neighbours carry less independent information than
    the plain Mann-Kendall test assumes. We remove the Theil-Sen trend, estimate
    the lag-1 autocorrelation of what is left (rho), and multiply the variance of S
    by (1 + rho) / (1 - rho), the factor for an AR(1) process. This is the
    large-sample form of the effective-sample-size correction of Yue & Wang (2004).

    What this is NOT: it is not the full modified Mann-Kendall test of
    Hamed & Rao (1998), which sums significant autocorrelations of the ranks over
    many lags, nor the trend-free pre-whitening procedure of Yue & Wang (2002).
    It uses only lag 1.

    break_year: if the step check found a strong jump, the leftovers are measured
    around the two flat levels instead of around a line. Otherwise a jump makes
    the leftovers look like slow, persistent noise, and the test would call a
    clear step "random variation" for the wrong reason.

    Three cautious choices:
      - rho gets a small-sample bias correction, (1 + 4 rho) / n, because lag-1
        estimates run low in short series;
      - rho is capped at 0.9, so the factor never exceeds 19;
      - the variance is never shrunk (factor >= 1), even when rho is negative.

    calibration.py measures the effect: on pure AR(1) noise with rho 0.3 and 0.5,
    Mizan calls about 6% of records Robust, where a plain straight-line fit calls
    16% and 27% of them significant.
    """
    if break_year is not None:
        after = t >= break_year
        r = x.copy()
        r[after] -= x[after].mean()
        r[~after] -= x[~after].mean()
    else:
        r = x - slope * (t - t.mean())
    return _inflation(r)


def _inflation(r: np.ndarray) -> float:
    """(1 + rho) / (1 - rho) for the lag-1 autocorrelation rho of the leftovers r."""
    n = len(r)
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


def trend_test(t, x, break_year: float | None = None) -> Trend:
    """
    Theil-Sen slope with a 95% range, and the corrected Mann-Kendall test.
    break_year: measure the autocorrelation around a step at this year (see above).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 years to test a trend.")

    slopes = np.sort(_pairwise_slopes(t, x))
    slope = float(np.median(slopes))
    intercept = float(np.median(x) - slope * np.median(t))

    s = _mk_s(x)
    factor = _autocorrelation_factor(t, x, slope, break_year)
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
                               # (computed on the effective number of independent years)


def step_check(t, x) -> Step:
    """
    Compare two explanations of the series:
        a straight line (gradual change), or
        one sudden jump between two flat levels.
    A jump can be a real shift in the climate, or a change in what the model was
    fed. Either way it is not a gradual trend, so the app reports it separately.

    BIC assumes independent years. Autocorrelated noise wanders slowly and can
    look like a jump, so the comparison uses the effective number of independent
    years, n / (1 + rho) * (1 - rho), with rho measured around the best step.
    The break position is searched over, so it counts as a parameter, and the
    margin of 6 ("strong" evidence) adds a further allowance for that search.
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

    r = x.copy()
    r[:best_k] -= x[:best_k].mean()
    r[best_k:] -= x[best_k:].mean()
    n_eff = n / _inflation(r)
    # BIC(line) - BIC(step): a line has 2 parameters, a step 3 (2 levels + the break position)
    margin = n_eff * math.log(rss_line / best_rss) - math.log(n_eff)
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


# ---------------------------------------------------------------- known change of data source

@dataclass
class SourceChange:
    year: float           # first year of the new data product
    bic_margin: float     # > 0 means a step at that year fits better than a straight line
    trend_inside: bool    # a significant same-direction trend exists on one side of the switch
    explains: bool        # the switch explains the change at least as well as a trend does


def source_change_check(t, x, year: float, min_side: int = 8) -> SourceChange | None:
    """
    Some records join two data products at a known year (NASA POWER sunlight:
    GEWEX SRB up to 2000, CERES from 2001). A jump at that year is more likely an
    artifact of the switch than a change in the climate.

    The step check above looks for the single best break anywhere, so a smaller
    jump at the switch can hide inside a record and be read as a gradual trend.
    Here the break is fixed at the switch: two flat levels, split at `year`,
    against a straight line. Both models have two parameters, so the better fit
    is simply the smaller residual sum of squares.

    The switch "explains" the change when the two-level model fits better AND
    neither side, on its own, shows a significant trend in the same direction.
    Returns None when either side has fewer than `min_side` years.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    after = t >= year
    if after.sum() < min_side or (~after).sum() < min_side:
        return None
    n = len(x)
    line = np.polyfit(t, x, 1)
    rss_line = float(((x - np.polyval(line, t)) ** 2).sum()) + 1e-12
    rss_step = float(((x[after] - x[after].mean()) ** 2).sum()
                     + ((x[~after] - x[~after].mean()) ** 2).sum()) + 1e-12
    margin = n * math.log(rss_line / rss_step)
    direction = np.sign(trend_test(t, x).slope)
    trend_inside = False
    for side in (~after, after):
        part = trend_test(t[side], x[side])
        if part.p < ALPHA and np.sign(part.slope) == direction:
            trend_inside = True
    return SourceChange(float(year), float(margin), trend_inside, margin > 0 and not trend_inside)


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
    """
    Re-run the trend test for every start and end year at least `shortest` years apart.

    Two lengths are involved:
      - `shortest` (15 years) is the shortest period that is computed and drawn
        in the app's stress-test chart.
      - `long_length` is the shortest period that counts toward `agreement`, the
        share that must point the same way as the full record (90% for Robust).
        Short periods flip direction easily, so they are drawn but not counted.

    The windows overlap heavily, so `agreement` is not a probability and 90% is
    not a confidence level. It is a stability screen against cherry-picked start
    and end years. Its effect on false alarms and detections is measured only by
    calibration.py. It also marks change that starts late in the record
    (flat, then rising) as unstable, even when that change is real.

    For records of 30 years or more, long_length is MIN_YEARS (20 years). Every
    built-in NASA POWER record qualifies: 45 years for weather, 42 for sunlight.
    A record of 20 to 29 years (only possible through "Any point on Earth" with a
    shorter series) cannot fit many 20-year periods, so long_length becomes 60% of
    the record, rounded down, and never under 10 years: 12 years for a 20-year
    record, 17 for a 29-year one. `shortest` is lowered to match when needed.
    Records under MIN_YEARS never reach this test; judge() returns Inconclusive first.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    full_sign = np.sign(trend_test(t, x).slope)
    # 20 years for records of 30+ years; 60% of the record (at least 10) for shorter ones
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
    source: SourceChange | None = None                # only for records with a known source change


def judge(t, x, known_break: float | None = None) -> Verdict:
    """
    Weigh the evidence and return one of: Robust, Inconclusive, Abrupt shift.

    known_break: first year of a new data product inside the record, if any
    (power.SOURCE_CHANGE). A change that the switch explains is never Robust.

    Notes (first one is the main reason):
      too_short, not_significant, step, unstable, source_change,
      and many_places (added by callers after Benjamini-Hochberg).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_YEARS:
        return Verdict(INCONCLUSIVE, notes=["too_short"])

    # The step check runs first. If the record is one strong jump, the noise test
    # measures the year-to-year noise around that jump, so a clear step is not
    # mistaken for slow random wandering.
    step = step_check(t, x)
    trend = trend_test(t, x, break_year=step.break_year if step.is_step else None)
    windows = window_test(t, x)
    source = None
    if known_break is not None and t[0] < known_break <= t[-1]:
        source = source_change_check(t, x, known_break)

    def verdict(label, notes):
        return Verdict(label, trend, step, windows, notes, source)

    if not trend.significant:
        return verdict(INCONCLUSIVE, ["not_significant"])
    if step.is_step:
        at_switch = known_break is not None and abs(step.break_year - known_break) <= 1
        return verdict(SHIFT, ["step", "source_change"] if at_switch else ["step"])
    if windows.agreement < AGREEMENT_NEEDED:
        return verdict(INCONCLUSIVE, ["unstable"])
    if source is not None and source.explains:
        return verdict(INCONCLUSIVE, ["source_change"])
    return verdict(ROBUST, [])


# ---------------------------------------------------------------- many places at once

def noise_p(t, x) -> float:
    """
    The noise-test p-value exactly as judge() computes it (step check first),
    without the slow stress test. Used to adjust across places quickly.
    """
    step = step_check(t, x)
    return trend_test(t, x, break_year=step.break_year if step.is_step else None).p


def benjamini_hochberg(p_values, q: float = ALPHA):
    """
    False discovery rate control. Returns (adjusted p-values, keep flags).

    Applied only when several places are tested together. The family is fixed in
    advance: the 15 built-in areas, for one variable and one season (35 families
    in scan.py). There is no correction across all 525 tests. A single-place
    query in "Investigate one place" is one test, so its p-value is not adjusted;
    the app shows the family-adjusted value next to it for the built-in areas.

    BH stays valid when the places are positively correlated, as neighbouring
    grid cells are, but 15 agreeing places are fewer independent pieces of
    evidence than the count suggests.

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


def adjust_places(p_values, series):
    """
    Benjamini-Hochberg across places, counting identical records once.

    Neighbouring places can fall in the same grid cell (NASA POWER sunlight uses
    1-degree cells, so Jeddah and Makkah share one record). Counting the same
    record twice would make the correction look stricter than it is, and would
    count one piece of evidence as two.

    series: one (years, values) pair per place, in the same order as p_values.
    Returns (adjusted p-values, keep flags), one per place.
    """
    keys = [(tuple(np.asarray(y, dtype=float)), tuple(np.round(np.asarray(v, dtype=float), 9)))
            for y, v in series]
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    unique = list(first.values())
    adjusted_u, keep_u = benjamini_hochberg([p_values[i] for i in unique])
    where = {key: k for k, key in enumerate(first)}
    adjusted = np.array([adjusted_u[where[key]] for key in keys])
    keep = np.array([keep_u[where[key]] for key in keys], dtype=bool)
    return adjusted, keep


def apply_places(verdicts, keep):
    """
    After Benjamini-Hochberg, a place keeps a significant verdict (Robust or
    Abrupt shift) only if its adjusted p-value passes. Otherwise it becomes
    Inconclusive with the note "many_places". Returns the new labels.
    """
    labels = []
    for v, k in zip(verdicts, keep):
        if v.label in (ROBUST, SHIFT) and not k:
            v.label = INCONCLUSIVE
            v.notes = ["many_places"] + v.notes
        labels.append(v.label)
    return labels
