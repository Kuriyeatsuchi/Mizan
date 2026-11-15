"""Checks that the statistics behave as claimed. Run with:  python -m pytest"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import power   # noqa: E402
import stats   # noqa: E402

YEARS = np.arange(1981, 2026, dtype=float)


def test_clear_trend_is_robust():
    rng = np.random.default_rng(1)
    x = 0.04 * (YEARS - YEARS[0]) + rng.normal(0, 0.3, len(YEARS))
    v = stats.judge(YEARS, x)
    assert v.label == stats.ROBUST
    assert v.trend.slope_low < 0.04 < v.trend.slope_high


def test_pure_noise_is_rarely_robust():
    rng = np.random.default_rng(2)
    robust = sum(stats.judge(YEARS, rng.normal(0, 1, len(YEARS))).label == stats.ROBUST for _ in range(200))
    assert robust / 200 < 0.06


def test_single_jump_is_flagged():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.2, len(YEARS))
    x[YEARS >= 2001] += 1.5
    v = stats.judge(YEARS, x)
    assert v.label == stats.SHIFT
    assert abs(v.step.break_year - 2001) <= 1


def test_short_record_is_inconclusive():
    v = stats.judge(YEARS[:10], np.arange(10.0))
    assert v.label == stats.INCONCLUSIVE and "too_short" in v.notes


def test_benjamini_hochberg_known_values():
    adjusted, keep = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(adjusted, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
    assert keep.tolist() == [True, False, False, False]


def test_parse_power_response():
    payload = {
        "header": {"fill_value": -999.0},
        "parameters": {"T2M": {"units": "C", "longname": "Temperature at 2 Meters"}},
        "properties": {"parameter": {"T2M": {
            "202001": 23.0, "202002": 24.0, "202012": 25.0, "202013": 24.0,   # 13 = annual value
            "202101": 23.5, "202102": -999.0,
        }}},
    }
    table, meta = power.parse(payload, "T2M")
    assert len(table) == 4 and meta["units"] == "C"
    assert 13 not in table["month"].tolist()


def test_winter_uses_december_of_previous_year():
    import pandas as pd
    table = pd.DataFrame({"year": [2020, 2021, 2021], "month": [12, 1, 2], "value": [10.0, 20.0, 30.0]})
    years, values = power.yearly(table, [12, 1, 2])
    assert years.tolist() == [2021.0] and values.tolist() == [20.0]


def test_saved_copy_is_used_without_internet(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(power, "DATA_DIR", tmp_path)
    table = pd.DataFrame({"year": [2020], "month": [1], "value": [23.0]})
    power.save_copy(21.54, 39.17, "T2M", table, {"units": "C", "longname": "x", "url": "u", "accessed": "2026-11-14"})
    monkeypatch.setattr(power.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    loaded, meta = power.fetch(21.54, 39.17, "T2M")
    assert loaded["value"].tolist() == [23.0] and meta["saved_copy"] is True


def test_naive_p_matches_a_clear_line():
    assert stats.naive_p(YEARS, 0.1 * YEARS) < 1e-6


def test_sources_are_attributed_per_variable():
    for weather in ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS10M"):
        assert "MERRA-2" in power.source_of(weather)
    sunlight = power.source_of("ALLSKY_SFC_SW_DWN")
    assert "GEWEX SRB" in sunlight and "CERES SYN1deg" in sunlight and "MERRA-2" not in sunlight
    assert set(power.SOURCES) == set(power.VARIABLES)


def test_saved_copies_carry_their_source():
    lat, lon = power.CITIES["Tabuk"]
    _, meta = power.fetch(lat, lon, "ALLSKY_SFC_SW_DWN")
    assert meta["saved_copy"] is True and "CERES" in meta["source"]


def test_stress_test_window_length():
    rng = np.random.default_rng(4)
    assert stats.window_test(YEARS, rng.normal(0, 1, 45)).long_length == 20          # 45-year record
    assert stats.window_test(YEARS[:42], rng.normal(0, 1, 42)).long_length == 20     # sunlight, 42 years
    assert stats.window_test(YEARS[:25], rng.normal(0, 1, 25)).long_length == 15     # 60% of 25
    assert stats.window_test(YEARS[:20], rng.normal(0, 1, 20)).long_length == 12     # 60% of 20


# ------------------------------------------------------------ reference values (SciPy)

def test_theil_sen_and_interval_match_scipy():
    from scipy.stats import theilslopes
    rng = np.random.default_rng(5)
    # alternating noise has negative lag-1 autocorrelation, so no correction is applied
    x = 0.02 * (YEARS - YEARS[0]) + 0.3 * (-1) ** np.arange(len(YEARS)) + rng.normal(0, 0.1, len(YEARS))
    t = stats.trend_test(YEARS, x)
    assert t.variance_factor == 1.0
    ref = theilslopes(x, YEARS, alpha=0.95)
    assert np.isclose(t.slope, ref.slope)
    assert np.isclose(t.slope_low, ref.low_slope) and np.isclose(t.slope_high, ref.high_slope)


def test_mann_kendall_s_matches_kendall_tau():
    from scipy.stats import kendalltau
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, len(YEARS)) + 0.01 * np.arange(len(YEARS))
    n = len(x)
    tau = kendalltau(YEARS, x).statistic
    assert np.isclose(stats.trend_test(YEARS, x).s, tau * n * (n - 1) / 2)


def test_benjamini_hochberg_matches_scipy():
    from scipy.stats import false_discovery_control
    p = np.random.default_rng(8).uniform(0, 0.2, 15)
    adjusted, _ = stats.benjamini_hochberg(p)
    assert np.allclose(adjusted, false_discovery_control(p))


# ------------------------------------------------------------ edge cases

def test_constant_series_does_not_crash():
    v = stats.judge(YEARS, np.full(len(YEARS), 5.0))
    assert v.label == stats.INCONCLUSIVE and v.trend.slope == 0.0


# ------------------------------------------------------------ fixes from the review round

def _record(city, parameter, season):
    table, _ = power.fetch(*power.CITIES[city], parameter)
    return power.yearly(table, power.SEASONS[season])


def test_strong_step_is_an_abrupt_shift_not_noise():
    # Tabuk sunlight: one large jump. Measured around a straight line the jump looks
    # like slow random wandering; the step check must run first.
    y, x = _record("Tabuk", "ALLSKY_SFC_SW_DWN", "Whole year")
    v = stats.judge(y, x, known_break=power.SOURCE_CHANGE["ALLSKY_SFC_SW_DWN"])
    assert v.step.is_step and v.label == stats.SHIFT


def test_step_on_autocorrelated_noise_uses_effective_years():
    # strongly autocorrelated noise wanders slowly; it should rarely be read as a step
    rng = np.random.default_rng(9)
    steps = 0
    for _ in range(100):
        x = np.empty(len(YEARS)); x[0] = rng.normal()
        for i in range(1, len(YEARS)):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, np.sqrt(1 - 0.49))
        steps += stats.judge(YEARS, x).label == stats.SHIFT
    assert steps <= 4


def test_sunlight_source_change_blocks_robust():
    # Dammam winter sunlight: two flat levels split at 2001 fit better than a trend,
    # and neither side has a trend of its own.
    y, x = _record("Dammam", "ALLSKY_SFC_SW_DWN", "Winter (Dec-Feb)")
    assert stats.judge(y, x).label == stats.ROBUST          # without the check
    v = stats.judge(y, x, known_break=2001)
    assert v.label == stats.INCONCLUSIVE and v.notes == ["source_change"]


def test_sunlight_trend_that_survives_the_switch_stays_robust():
    y, x = _record("Jazan", "ALLSKY_SFC_SW_DWN", "Summer (Jun-Aug)")
    v = stats.judge(y, x, known_break=2001)
    assert v.label == stats.ROBUST and not v.source.explains


def test_identical_records_count_once_in_benjamini_hochberg():
    a = (YEARS, np.arange(45.0)); b = (YEARS, np.arange(45.0) * 2)
    adjusted, _ = stats.adjust_places([0.02, 0.02, 0.04], [a, a, b])
    assert np.allclose(adjusted, [0.04, 0.04, 0.04])          # 2 distinct records, not 3


def test_shift_verdicts_are_adjusted_too():
    v = stats.Verdict(stats.SHIFT, notes=["step"])
    stats.apply_places([v], np.array([False]))
    assert v.label == stats.INCONCLUSIVE and v.notes[0] == "many_places"


def test_app_never_saves_custom_points(tmp_path, monkeypatch):
    monkeypatch.setattr(power, "DATA_DIR", tmp_path)
    payload = {"parameters": {"T2M": {"units": "C", "longname": "x"}},
               "properties": {"parameter": {"T2M": {"202001": 23.0}}}}

    class Reply:
        ok, status_code, text = True, 200, ""
        def json(self): return payload

    monkeypatch.setattr(power.requests, "get", lambda *a, **k: Reply())
    power.fetch(1.0, 2.0, "T2M")
    assert list(tmp_path.iterdir()) == []
    power.fetch(1.0, 2.0, "T2M", save=True)
    assert len(list(tmp_path.iterdir())) == 2


def test_committed_results_match_the_code():
    # results/all_results.csv must be regenerated (python scan.py) whenever the method changes
    import pandas as pd
    import scan
    saved = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "results", "all_results.csv"))
    saved = saved[(saved["param"] == "T2M") & (saved["season"] == "Whole year")].set_index("city")
    rows, errors = scan.judge_places("T2M", "Whole year")
    assert not errors and len(rows) == len(saved)
    for row in rows:
        assert row["verdict"] == saved.loc[row["city"], "verdict"]
        assert np.isclose(row["p"], saved.loc[row["city"], "p"], rtol=1e-9, atol=0)
