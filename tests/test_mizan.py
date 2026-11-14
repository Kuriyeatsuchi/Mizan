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
