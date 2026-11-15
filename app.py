"""
Mizan: is the trend real?

A Streamlit app for the NASA Space Apps 2026 challenge "Be An Earth System Trend Detective!".
Pick a place and a NASA variable. Mizan weighs the evidence and returns a verdict.

Run locally:   streamlit run app.py

AI assistance: this file was drafted with Claude (Anthropic) during the hackathon
and reviewed and tested by the team. See README.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import power
import stats

BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
VERDICT_STYLE = {   # symbol + colour, so the verdict never depends on colour alone
    stats.ROBUST: ("✔", "#0a7a0a"),
    stats.INCONCLUSIVE: ("?", "#6b6b68"),
    stats.SHIFT: ("⚠", "#b06a00"),
}

st.set_page_config(page_title="Mizan: is the trend real?", page_icon="⚖️", layout="wide")


# ------------------------------------------------------------------ helpers

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load(lat: float, lon: float, parameter: str):
    return power.fetch(lat, lon, parameter)


def unit_text(units: str) -> str:
    return {"C": "°C"}.get(units, units)


def explain(v: stats.Verdict, unit: str) -> list[str]:
    """Plain-language reasons for a verdict."""
    if "too_short" in v.notes:
        return [f"The record has fewer than {stats.MIN_YEARS} complete years, which is too short to judge a trend."]
    t, w, s = v.trend, v.windows, v.step
    per_decade = t.slope * 10
    direction = "rising" if per_decade > 0 else "falling"
    lines = [
        f"Size: {direction} by {abs(per_decade):.2f} {unit} per decade "
        f"(95% range {t.slope_low * 10:+.2f} to {t.slope_high * 10:+.2f})."
    ]
    if t.significant:
        chance = "less than 0.1%" if t.p < 0.001 else f"about {t.p * 100:.1f}%"
        p_text = "p < 0.001" if t.p < 0.001 else f"p = {t.p:.3f}"
        lines.append(f"Noise test: passed. A change this consistent would appear by chance {chance} of the time ({p_text}).")
    else:
        lines.append(f"Noise test: failed. Random year-to-year variation could produce this pattern "
                     f"(p = {t.p:.2f}, needs to be below {stats.ALPHA}).")
    if t.variance_factor > 1.05:
        lines.append(f"Neighbouring years resemble each other here, so the test was made "
                     f"{t.variance_factor:.1f}x stricter to compensate.")
    lines.append(f"Stress test: {w.agreement * 100:.0f}% of all periods of {w.long_length}+ years point the same way "
                 f"as the full record (needs {stats.AGREEMENT_NEEDED * 100:.0f}%).")
    if s.is_step:
        lines.append(f"Shift check: one shift of {s.jump:+.2f} {unit} around {int(s.break_year)} explains the "
                     f"data better than a gradual trend. It may be a real shift in the climate or a change in "
                     f"what the model was fed. Compare with another record before calling it a trend.")
    else:
        lines.append("Shift check: passed. The change is spread over the record, not concentrated in one step.")
    return lines


def gotcha(v: stats.Verdict, years, values) -> str | None:
    """A sentence for the cases where a spreadsheet trend line and Mizan disagree."""
    if v.trend is None:
        return None
    plain = stats.naive_p(years, values)
    if plain < stats.ALPHA and v.label != stats.ROBUST:
        why = {"not_significant": "it fails the noise test once we account for years resembling their neighbours",
               "unstable": "its direction depends on which years you pick",
               "step": "the change is one abrupt shift, not a gradual trend"}[v.notes[0]]
        shown = "p < 0.001" if plain < 0.001 else f"p = {plain:.3f}"
        return f"A plain trend line calls this significant ({shown}). Mizan does not, because {why}."
    return None


def verdict_card(v: stats.Verdict):
    symbol, colour = VERDICT_STYLE[v.label]
    st.markdown(
        f"<div style='border-left:8px solid {colour};padding:0.6rem 1rem;margin:0.5rem 0 1rem;'>"
        f"<div style='font-size:0.85rem;opacity:0.7;'>VERDICT</div>"
        f"<div style='font-size:2rem;font-weight:700;line-height:1.2;'>{symbol} {v.label}</div></div>",
        unsafe_allow_html=True,
    )


def series_chart(years, values, v: stats.Verdict, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    if v.trend is not None:
        t = v.trend
        mid_t, mid_x = np.median(years), np.median(values)
        a = mid_x + t.slope_low * (years - mid_t)
        b = mid_x + t.slope_high * (years - mid_t)
        fig.add_trace(go.Scatter(x=years, y=np.maximum(a, b), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=np.minimum(a, b), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(235,104,52,0.18)",
                                 hoverinfo="skip", name="95% range of the trend"))
        fig.add_trace(go.Scatter(x=years, y=t.intercept + t.slope * years, mode="lines",
                                 line=dict(color=ORANGE, width=2.5), name="Trend (Theil-Sen)",
                                 hovertemplate="%{y:.2f} " + unit + "<extra>Trend</extra>"))
    fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers",
                             line=dict(color=BLUE, width=1.5), marker=dict(size=7, color=BLUE),
                             name="Yearly value",
                             hovertemplate="%{y:.2f} " + unit + "<extra>%{x:.0f}</extra>"))
    if v.step is not None and v.step.is_step:
        fig.add_vline(x=v.step.break_year - 0.5, line_dash="dash", line_color="#b06a00",
                      annotation_text=f"possible jump, {int(v.step.break_year)}", annotation_position="top")
    fig.update_layout(title=f"{title} ({unit})", yaxis_title=None, xaxis_title=None, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def window_chart(w: stats.Windows, unit: str) -> go.Figure:
    starts, ends = np.unique(w.start), np.unique(w.end)
    grid = np.full((len(ends), len(starts)), np.nan)
    grid[np.searchsorted(ends, w.end), np.searchsorted(starts, w.start)] = w.slope * 10
    limit = float(np.nanmax(np.abs(grid))) or 1.0
    fig = go.Figure(go.Heatmap(
        x=starts, y=ends, z=grid, zmin=-limit, zmax=limit, zmid=0,
        colorscale=[[0, BLUE], [0.5, NEUTRAL], [1, RED]],
        colorbar=dict(title=f"{unit} per decade"),
        hovertemplate="%{x:.0f} to %{y:.0f}: %{z:+.2f} " + unit + " per decade<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=w.start[w.significant], y=w.end[w.significant], mode="markers",
                             marker=dict(size=3, color="#33332f"), hoverinfo="skip",
                             name="passes the noise test"))
    fig.update_layout(title="Stress test: the trend for every start and end year",
                      xaxis_title="Start year", yaxis_title="End year",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10), height=440)
    return fig


def cities_chart(table: pd.DataFrame, unit: str) -> go.Figure:
    table = table.sort_values("Change per decade")
    symbols = {stats.ROBUST: "circle", stats.INCONCLUSIVE: "circle-open", stats.SHIFT: "diamond"}
    fig = go.Figure()
    for label, part in table.groupby("Verdict", sort=False):
        fig.add_trace(go.Scatter(
            x=part["Change per decade"], y=part["City"], mode="markers", name=label,
            marker=dict(size=11, color=VERDICT_STYLE[label][1], symbol=symbols[label], line=dict(width=2)),
            error_x=dict(type="data", symmetric=False, thickness=1.5, width=0,
                         array=part["High"] - part["Change per decade"],
                         arrayminus=part["Change per decade"] - part["Low"]),
            hovertemplate="%{y}: %{x:+.2f} " + unit + " per decade<extra>" + label + "</extra>"))
    fig.add_vline(x=0, line_color="#8a8a86", line_width=1)
    fig.update_layout(title=f"Change per decade with 95% range ({unit})",
                      yaxis=dict(categoryorder="array", categoryarray=table["City"].tolist()),
                      legend=dict(orientation="h", y=-0.12), margin=dict(l=10, r=10, t=50, b=10),
                      height=120 + 28 * len(table))
    return fig


LABEL_SIDE = {"Jeddah": "middle left", "Makkah": "bottom center", "Taif": "middle right", "Yanbu": "middle left",
              "Madinah": "top right", "Dammam": "top right", "Al-Ahsa": "bottom right", "Abha": "top left",
              "Jazan": "bottom left", "Najran": "middle right"}


def cities_map(table: pd.DataFrame, unit: str) -> go.Figure:
    symbols = {stats.ROBUST: "circle", stats.INCONCLUSIVE: "circle-open", stats.SHIFT: "diamond"}
    limit = float(table["Change per decade"].abs().max()) or 1.0
    fig = go.Figure()
    for i, (label, part) in enumerate(table.groupby("Verdict", sort=False)):
        fig.add_trace(go.Scattergeo(
            lon=part["lon"], lat=part["lat"], text=part["City"], mode="markers+text",
            textposition=[LABEL_SIDE.get(c, "top center") for c in part["City"]],
            name=label, customdata=part["Change per decade"],
            marker=dict(size=13, symbol=symbols[label], color=part["Change per decade"],
                        colorscale=[[0, BLUE], [0.5, "#b9b8b2"], [1, RED]], cmin=-limit, cmax=limit,
                        line=dict(width=2), showscale=(i == 0),
                        colorbar=dict(title=f"{unit} per decade", len=0.7)),
            hovertemplate="%{text}: %{customdata:+.2f} " + unit + " per decade<extra>" + label + "</extra>"))
    fig.update_geos(lonaxis_range=[33, 57], lataxis_range=[14.5, 33.5], projection_type="mercator",
                    showcountries=True, countrycolor="#8a8a86", showland=True, landcolor="rgba(128,128,128,0.12)",
                    showocean=True, oceancolor="rgba(42,120,214,0.08)", resolution=50, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(title="Where: filled = robust, hollow = inconclusive, diamond = abrupt shift",
                      legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=50, b=10), height=520)
    return fig


# ------------------------------------------------------------------ sidebar

st.sidebar.title("⚖️ Mizan")
st.sidebar.caption("Is the trend real? Weighing the evidence in NASA data.")

parameter = st.sidebar.selectbox("Variable", list(power.VARIABLES), format_func=lambda k: power.VARIABLES[k])
season = st.sidebar.selectbox("Months", list(power.SEASONS))
place_mode = st.sidebar.radio("Place", ["Saudi city", "Any point on Earth"], horizontal=True)
if place_mode == "Saudi city":
    city = st.sidebar.selectbox("City", list(power.CITIES))
    lat, lon = power.CITIES[city]
    place = f"the {city} area"
else:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 21.54, step=0.5)
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 39.17, step=0.5)
    place = f"the area around {lat:.2f}, {lon:.2f}"
months = power.SEASONS[season]

tab_one, tab_many, tab_method = st.tabs(["Investigate one place", "Compare Saudi cities", "How Mizan decides"])


# ------------------------------------------------------------------ one place

with tab_one:
    st.header(f"{power.VARIABLES[parameter]} in {place}")
    try:
        with st.spinner("Asking NASA POWER for the data..."):
            table, meta = load(lat, lon, parameter)
    except Exception as err:   # show the problem instead of a stack trace
        st.error(str(err))
        st.stop()

    unit = unit_text(meta["units"])
    years, values = power.yearly(table, months)
    if len(years) < 4:
        st.warning("NASA POWER has too few complete years of this variable here.")
        st.stop()

    st.caption("One NASA POWER grid cell, roughly 50 km wide. Near a coast it mixes land and sea, "
               "and it cannot see a city's own heat island.")
    if parameter == "PRECTOTCORR":
        st.warning("Desert rainfall is mostly zeros with a few big storms, and model rainfall is weak over deserts. "
                   "Treat rainfall verdicts here with extra caution.")

    if parameter == "ALLSKY_SFC_SW_DWN":
        st.warning("This sunlight record joins two NASA products: GEWEX SRB up to 2000 and CERES from 2001. "
                   "NASA advises caution when reading climate trends across a change of source.")

    verdict = stats.judge(years, values)
    disagreement = gotcha(verdict, years, values)
    if disagreement:
        st.info(disagreement)
    left, right = st.columns([1, 2])
    with left:
        verdict_card(verdict)
        for line in explain(verdict, unit):
            st.markdown(f"- {line}")
    with right:
        st.plotly_chart(series_chart(years, values, verdict, f"{season}, {int(years[0])} to {int(years[-1])}", unit),
                        key="series")

    if verdict.windows is not None:
        st.plotly_chart(window_chart(verdict.windows, unit), key="windows")
        st.caption("Each cell is one possible study period. Red means rising, blue falling, grey no change. "
                   "A dot marks periods that pass the noise test on their own. A trustworthy trend keeps one "
                   "colour across the chart. If the colour flips, the answer depends on the years you pick.")

    with st.expander("Data source"):
        st.markdown(
            f"**NASA POWER**, parameter `{parameter}` ({meta['longname'] or power.VARIABLES[parameter]}), "
            f"monthly values averaged over the chosen months.\n\n"
            f"Source product: {meta.get('source') or power.source_of(parameter)} "
            f"([NASA POWER data sources]({power.SOURCES_URL}))\n\n"
            f"Request: `{meta['url']}`\n\nAccessed: {meta['accessed']}"
            + ("  (saved copy in this repository)" if meta.get("saved_copy") else ""))
        st.download_button("Download the yearly values (CSV)",
                           pd.DataFrame({"year": years.astype(int), "value": values}).to_csv(index=False),
                           file_name=f"mizan_{place}_{parameter}.csv".replace(" ", "_"))


# ------------------------------------------------------------------ many places

with tab_many:
    st.header(f"{power.VARIABLES[parameter]} across Saudi Arabia")
    st.write(f"Runs the same checks for {len(power.CITIES)} cities ({season.lower()}), then corrects for the fact that "
             "testing many places makes false alarms more likely.")
    if st.button("Run all cities"):
        rows, unit_all = [], ""
        progress = st.progress(0.0)
        for i, (name, (clat, clon)) in enumerate(power.CITIES.items()):
            try:
                ctable, cmeta = load(clat, clon, parameter)
                unit_all = unit_text(cmeta["units"])
                cyears, cvalues = power.yearly(ctable, months)
                v = stats.judge(cyears, cvalues)
                if v.trend is not None:
                    plain = stats.naive_p(cyears, cvalues)
                    rows.append(dict(City=name, Verdict=v.label, p=v.trend.p, lat=clat, lon=clon,
                                     **{"Plain trend line": "significant" if plain < stats.ALPHA else "not significant"},
                                     **{"Change per decade": v.trend.slope * 10,
                                        "Low": v.trend.slope_low * 10, "High": v.trend.slope_high * 10}))
            except Exception as err:
                st.warning(f"{name}: {err}")
            progress.progress((i + 1) / len(power.CITIES))
        progress.empty()

        if rows:
            result = pd.DataFrame(rows)
            # Benjamini-Hochberg false discovery rate correction, applied HERE ONLY:
            # this tab tests all 15 areas at once for one variable and season, so the
            # chance of a false alarm somewhere is higher than for one test. The
            # "Investigate one place" tab runs a single test and is not adjusted.
            adjusted, keep = stats.benjamini_hochberg(result["p"].to_numpy())
            result["Adjusted p"] = adjusted
            # A city only stays "Robust" if it survives the many-places correction.
            result.loc[(result["Verdict"] == stats.ROBUST) & ~keep, "Verdict"] = stats.INCONCLUSIVE
            st.session_state["cities"] = (result, unit_all, parameter, season)

    if "cities" in st.session_state and st.session_state["cities"][2:] == (parameter, season):
        result, unit_all, _, _ = st.session_state["cities"]
        map_col, chart_col = st.columns(2)
        with map_col:
            st.plotly_chart(cities_map(result, unit_all), key="map")
        with chart_col:
            st.plotly_chart(cities_chart(result, unit_all), key="cities")
        shown = result[["City", "Verdict", "Plain trend line", "Change per decade", "Low", "High", "p", "Adjusted p"]].round(3)
        for column in ("p", "Adjusted p"):
            shown[column] = result[column].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
        st.dataframe(shown.sort_values("Change per decade", ascending=False), hide_index=True)

        robust = result[result["Verdict"] == stats.ROBUST]
        up = robust[robust["Change per decade"] > 0].sort_values("Change per decade", ascending=False)
        down = robust[robust["Change per decade"] < 0].sort_values("Change per decade")
        if len(up) and len(down):
            st.success(f"Opposite twins: {up.iloc[0]['City']} is rising ({up.iloc[0]['Change per decade']:+.2f} {unit_all} "
                       f"per decade) while {down.iloc[0]['City']} is falling ({down.iloc[0]['Change per decade']:+.2f}). "
                       "Both pass every check.")
        else:
            st.info("No opposite twins here: no two cities have robust trends in opposite directions.")
        fooled = result[(result["Plain trend line"] == "significant") & (result["Verdict"] != stats.ROBUST)]
        if len(fooled):
            st.warning("A plain trend line would call these significant, and Mizan does not: "
                       + ", ".join(fooled["City"]) + ". Open each one in the first tab to see why.")
        st.download_button("Download this table (CSV)", shown.to_csv(index=False),
                           file_name=f"mizan_cities_{parameter}.csv")


# ------------------------------------------------------------------ method

with tab_method:
    st.header("How Mizan decides")
    st.markdown(f"""
Mizan means “balance” in Arabic. A rising line is easy to draw. Mizan weighs four pieces of evidence before it calls that line a trend.

**The decision rule**: the same four checks, in the same order and with the same thresholds, for every place, variable and season.

| Check | Question | Passes when |
|---|---|---|
| Record length | Is there enough data? | At least {stats.MIN_YEARS} complete years |
| Noise test | Could random variation explain it? | Mann-Kendall p below {stats.ALPHA} after a lag-1 autocorrelation correction, and the 95% range of the slope excludes zero |
| Shift check | Is it gradual? | A single step does not fit clearly better than a line (BIC difference of {stats.STEP_BIC_MARGIN:.0f} or less), or a significant same-direction trend exists on one side of the step |
| Stress test | Does it survive other start and end years? | At least {stats.AGREEMENT_NEEDED * 100:.0f}% of all periods of {stats.MIN_YEARS}+ years have the same direction as the full record (for records under 30 years: periods of at least 60% of the record) |

**Verdicts**

- **Robust:** passes all four. Statistics cannot prove a trend, so we do not use that word.
- **Inconclusive:** fails the noise test or the stress test. This does not mean nothing is changing. It means this record cannot show it.
- **Abrupt shift:** the change is significant but concentrated in one step. It may be a real shift or a change in what the model was fed, so it should be compared with another record.

**Why these checks**

- *Size.* The Theil-Sen slope is the median of the slopes between every pair of years. One freak year cannot drag it.
- *Noise test.* Years that resemble their neighbours carry less independent information, so the test is made stricter when we detect that. We use a lag-1 variance inflation approximation, (1 + ρ) / (1 − ρ), the simplest form of the idea behind Hamed & Rao (1998) and Yue & Wang (2004), not their full methods.
- *Stress test.* Periods shorter than {stats.MIN_YEARS} years flip direction easily, so only longer ones count. This guards against cherry-picked start years.
- *Shift check.* NASA POWER's weather values come from a model (MERRA-2) that absorbed new satellites over time, and its sunlight record switches from GEWEX SRB to CERES in 2001. Both can create steps.
- *Many places.* In the *Compare Saudi cities* tab only, we adjust the p-values (Benjamini-Hochberg), because testing 15 places at the 5% level produces false alarms by chance. A single place in the first tab is one test and is not adjusted.

**What Mizan adds to a plain trend line.** A spreadsheet trend line assumes every year is independent and uses every start year you give it.
Mizan reports when the two disagree.

**How well calibrated is Mizan?** We ran Mizan, and an ordinary straight-line fit, on simulated 45-year records where
the truth is known (`calibration.py`, 300 records per case). Trends have year-to-year autocorrelation 0.3; steps happen in 2001.

| Simulated truth | Plain line significant | Mizan: Robust | Mizan: Abrupt shift |
|---|---|---|---|
| Pure noise, independent years | 4% | 3% | 0% |
| Pure noise, autocorrelation 0.3 | 16% | 6% | 0.3% |
| Pure noise, autocorrelation 0.5 | 27% | 6% | 0.7% |
| Weak trend (0.3 sd per decade) | 66% | 24% | 4% |
| Moderate trend (0.6 sd per decade) | 99% | 73% | 9% |
| Clear trend (1.0 sd per decade) | 100% | 93% | 6% |
| Small single step (1 sd) | 76% | 50% | 5% |
| Large single step (2 sd) | 100% | 39% | 59% |

On noise where neighbouring years resemble each other, as they do in climate data, a plain line raises a false alarm
up to 27% of the time; Mizan stays near 6%. The cost: Mizan misses most weak trends, and it cannot always tell a step
from a gradual change (it calls a small step Robust half the time).

**Limits**

- One grid cell of NASA POWER is roughly 50 km wide. Values describe an area, not a city centre or a mountain top.
- Rainfall verdicts in deserts are unreliable: mostly zeros, rare storms, and weak model rainfall.
- Records start in 1981 (1984 for sunlight). Mizan says nothing about earlier decades.
- The sunlight record joins GEWEX SRB (to 2000) and CERES (from 2001), so sunlight verdicts for whole records deserve caution.
- A robust trend says *that* something changed, not *why*.
""")

st.sidebar.divider()
st.sidebar.caption("Data: NASA POWER (NASA Langley Research Center). Built for the NASA Space Apps Challenge 2026. "
                   "Parts of the code were written with AI assistance (Claude); see the README.")
