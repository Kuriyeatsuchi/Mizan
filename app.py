"""
Mizan: is the trend real?

A Streamlit app for the NASA Space Apps 2026 challenge "Be An Earth System Trend Detective!".
Pick a place and a NASA variable. Mizan weighs the evidence and returns a verdict.

Run locally:   streamlit run app.py

AI assistance: this file was drafted with Claude (Anthropic) during the hackathon
and reviewed and tested by the team. See README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import power
import scan
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def family_adjusted(parameter: str, season: str) -> dict:
    """
    Benjamini-Hochberg across the built-in areas for one variable and season,
    exactly as the Compare tab and scan.py do it. Returns {area: (adjusted p, passes)}.
    """
    names, ps, series = [], [], []
    for name, (clat, clon) in power.CITIES.items():
        ctable, _ = load(clat, clon, parameter)
        y, x = power.yearly(ctable, power.SEASONS[season])
        if len(x) >= stats.MIN_YEARS:
            names.append(name)
            ps.append(stats.noise_p(y, x))
            series.append((y, x))
    adjusted, keep = stats.adjust_places(ps, series)
    return {n: (float(a), bool(k)) for n, a, k in zip(names, adjusted, keep)}


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def explain(v: stats.Verdict, unit: str, switch: str | None = None) -> list[str]:
    """Plain-language reasons for a verdict. switch names the change of data product, if any."""
    if "too_short" in v.notes:
        return [f"The record has fewer than {stats.MIN_YEARS} complete years, which is too short to judge a trend."]
    t, w, s, src = v.trend, v.windows, v.step, v.source
    per_decade = t.slope * 10
    direction = "rising" if per_decade > 0 else "falling"
    lines = [
        f"Size: {direction} by {abs(per_decade):.2f} {unit} per decade "
        f"(95% range of the slope {t.slope_low * 10:+.2f} to {t.slope_high * 10:+.2f})."
    ]
    if t.significant:
        chance = "less than 0.1%" if t.p < 0.001 else f"about {t.p * 100:.1f}%"
        lines.append(f"Noise test: passed. If nothing were changing, random variation would produce a pattern "
                     f"this consistent {chance} of the time ({p_text(t.p)}).")
    elif t.p < stats.ALPHA:
        lines.append(f"Noise test: failed. The p-value is small ({p_text(t.p)}), but the 95% range of the slope "
                     f"includes zero, so the size of the change cannot be told apart from no change. This happens "
                     f"when most values are equal, as in desert rainfall.")
    else:
        lines.append(f"Noise test: failed. If nothing were changing, random year-to-year variation would produce "
                     f"a pattern like this {t.p * 100:.0f}% of the time (p = {t.p:.2f}; Mizan needs below "
                     f"{stats.ALPHA * 100:.0f}%).")
    if t.variance_factor > 1.05:
        if s.is_step:
            lines.append(f"The year-to-year noise was measured around the jump, not around a straight line. "
                         f"Neighbouring years still resemble each other, so the test was made "
                         f"{t.variance_factor:.1f}x stricter.")
        else:
            lines.append(f"Neighbouring years resemble each other here (slow ups and downs, or a jump too small "
                         f"to single out), so the test was made {t.variance_factor:.1f}x stricter to compensate.")
    lines.append(f"Stress test: {w.agreement * 100:.0f}% of all periods of {w.long_length}+ years point the same way "
                 f"as the full record (needs {stats.AGREEMENT_NEEDED * 100:.0f}%). This is a stability check, "
                 f"not a probability.")
    if s.is_step and v.label == stats.SHIFT:
        lines.append(f"Shift check: one jump of {s.jump:+.2f} {unit} around {int(s.break_year)} explains the data "
                     f"better than a gradual trend. It may be a real shift in the climate, or a change in the data "
                     f"record (for example new satellites fed to the model). Compare with another record before "
                     f"reading it as climate.")
    elif s.is_step:
        lines.append(f"Shift check: one jump of {s.jump:+.2f} {unit} around {int(s.break_year)} fits better than a "
                     f"straight line, but the change is not clear of year-to-year noise. Treat it as a possible "
                     f"break in the record, not as a finding.")
    else:
        lines.append("Shift check: passed. The change is spread over the record, not concentrated in one step.")
    if src is not None and switch:
        if src.explains:
            lines.append(f"Source check: this record switches data product in {int(src.year)} ({switch}). Two flat "
                         f"levels split at that year fit as well as a trend, and neither side shows a trend of its "
                         f"own, so the change may be an artifact of the switch.")
        else:
            lines.append(f"Source check: passed. The change is not explained by the {int(src.year)} switch of data "
                         f"product ({switch}).")
    if "source_change" in v.notes and v.label == stats.SHIFT:
        lines.append("The jump falls within a year of that switch, so it is most likely an artifact of the record.")
    return lines


def gotcha(v: stats.Verdict, years, values) -> str | None:
    """A sentence for the cases where a spreadsheet trend line and Mizan disagree."""
    if v.trend is None:
        return None
    plain = stats.naive_p(years, values)
    if plain >= stats.ALPHA or v.label == stats.ROBUST:
        return None
    if v.label == stats.SHIFT:
        return (f"A plain trend line calls this a significant trend ({p_text(plain)}). Mizan agrees that something "
                f"changed, but finds one abrupt jump rather than a gradual trend.")
    why = {"not_significant": "it does not pass the noise test once neighbouring years are allowed to resemble each other",
           "unstable": "its direction depends on which years you pick",
           "source_change": "the change can be explained by the record's switch of data product",
           "many_places": "it does not survive the correction for testing many places at once",
           }.get(v.notes[0], "it fails one of the checks")
    return f"A plain trend line calls this significant ({p_text(plain)}). Mizan does not call it a trend, because {why}."


def verdict_card(v: stats.Verdict):
    symbol, colour = VERDICT_STYLE[v.label]
    st.markdown(
        f"<div style='border-left:8px solid {colour};padding:0.6rem 1rem;margin:0.5rem 0 1rem;'>"
        f"<div style='font-size:0.85rem;opacity:0.7;'>VERDICT</div>"
        f"<div style='font-size:2rem;font-weight:700;line-height:1.2;'>{symbol} {v.label}</div></div>",
        unsafe_allow_html=True,
    )


def series_chart(years, values, v: stats.Verdict, title: str, unit: str,
                 known_break: float | None = None) -> go.Figure:
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
                                 hoverinfo="skip", name="Range of plausible slopes (95%)"))
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
    if known_break is not None and years[0] < known_break <= years[-1]:
        fig.add_vline(x=known_break - 0.5, line_dash="dot", line_color="#8a8a86",
                      annotation_text=f"data product switch, {int(known_break)}", annotation_position="bottom right")
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
    """Change per decade with its 95% range, one row per area."""
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
    fig.update_layout(title=f"Change per decade, with 95% range ({unit})",
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
    fig.update_layout(title="Verdict by area",
                      legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=50, b=10), height=520)
    return fig


# ------------------------------------------------------------------ sidebar

# Worked examples: one click sets the sidebar. (variable, months, area, what to look for)
EXAMPLES = {
    "A robust trend": ("T2M", "Whole year", "Jeddah",
                       "Every check passes: the warming is steady, clear of noise, and survives other start and end "
                       "years. This is what Mizan calls Robust."),
    "A jump, not a trend": ("T2M", "Whole year", "Abha",
                            "A plain trend line sees steady warming. Mizan finds most of the change in one jump around "
                            "1998. Is that the climate or the record? Mizan cannot say, and tells you to check."),
    "A trap for trend lines": ("ALLSKY_SFC_SW_DWN", "Winter (Dec-Feb)", "Dammam",
                               "A spreadsheet trend line calls this rise in winter sunlight significant. But the "
                               "record switches satellite product in 2001 (dotted line). Two flat levels split at "
                               "that year fit as well as a trend, and neither side rises on its own, so the "
                               "\"trend\" may be the switch, not the sky."),
}
PLACE_MODES = ["Saudi area", "Any point on Earth"]


def use_example(name: str):
    parameter, season, city, _ = EXAMPLES[name]
    st.session_state.update(parameter=parameter, season=season, place_mode=PLACE_MODES[0], city=city)


st.sidebar.title("⚖️ Mizan")
st.sidebar.caption("Is the trend real? Weighing the evidence in NASA data.")

parameter = st.sidebar.selectbox("Variable", list(power.VARIABLES), format_func=lambda k: power.VARIABLES[k],
                                 key="parameter")
season = st.sidebar.selectbox("Months", list(power.SEASONS), key="season")
place_mode = st.sidebar.radio("Place", PLACE_MODES, horizontal=True, key="place_mode")
if place_mode == PLACE_MODES[0]:
    city = st.sidebar.selectbox("Area", list(power.CITIES), key="city")
    lat, lon = power.CITIES[city]
    place = f"the {city} area"
else:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 21.54, step=0.5)
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 39.17, step=0.5)
    place = f"the area around {lat:.2f}, {lon:.2f}"
months = power.SEASONS[season]
known_break = power.SOURCE_CHANGE.get(parameter)

st.sidebar.divider()
st.sidebar.markdown("**New here? Try a worked example**")
for name in EXAMPLES:
    st.sidebar.button(name, on_click=use_example, args=(name,), width="stretch")

tab_one, tab_many, tab_method = st.tabs(["Investigate one place", "Compare Saudi areas", "How Mizan decides"])


# ------------------------------------------------------------------ one place

with tab_one:
    st.header(f"{power.VARIABLES[parameter]} in {place}")
    st.caption(power.DESCRIPTIONS[parameter])
    for name, (e_param, e_season, e_city, note) in EXAMPLES.items():
        if place_mode == PLACE_MODES[0] and (parameter, season, city) == (e_param, e_season, e_city):
            st.info(f"**Worked example: {name.lower()}.** {note}")
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

    st.caption(f"Each value is {power.RESOLUTION[parameter]}. It describes an area, not a street: near a coast "
               "it mixes land and sea, and it cannot see a city's own heat island.")
    if parameter == "PRECTOTCORR":
        st.warning("Desert rainfall is mostly zeros with a few big storms, and model rainfall is weak over deserts. "
                   "Treat rainfall verdicts here with extra caution.")

    if parameter == "ALLSKY_SFC_SW_DWN":
        st.warning("This sunlight record joins two NASA products: GEWEX SRB up to 2000 and CERES from 2001. "
                   "NASA advises caution when reading climate trends across a change of source.")

    verdict = stats.judge(years, values, known_break=known_break)
    disagreement = gotcha(verdict, years, values)
    if disagreement:
        st.info(disagreement)
    reasons = explain(verdict, unit, power.SOURCE_CHANGE_NAME.get(parameter))
    if place_mode == PLACE_MODES[0] and verdict.trend is not None:
        family = family_adjusted(parameter, season).get(city)
        if family is not None:
            adjusted_p, passes = family
            if verdict.label in (stats.ROBUST, stats.SHIFT) and not passes:
                st.warning(f"On its own this record is **{verdict.label}**. Tested together with the other "
                           f"{len(power.CITIES) - 1} built-in areas, as in the *Compare Saudi areas* tab, its "
                           f"adjusted p-value is {adjusted_p:.3f}, so there it is **Inconclusive**: when many places "
                           f"are tested at once, one passing by chance becomes much more likely.")
            else:
                reasons.append(f"Many places: adjusted for testing all {len(power.CITIES)} built-in areas at once, "
                               f"{p_text(adjusted_p)}.")
    left, right = st.columns([1, 2])
    with left:
        verdict_card(verdict)
        for line in reasons:
            st.markdown(f"- {line}")
    with right:
        st.plotly_chart(series_chart(years, values, verdict, f"{season}, {int(years[0])} to {int(years[-1])}", unit,
                                     known_break), key="series")
        st.caption("The orange line is the Theil-Sen trend. The shaded wedge shows the range of plausible slopes "
                   "(95%), pivoting on the middle year. It is not a range for individual years' values.")

    if verdict.windows is not None:
        st.plotly_chart(window_chart(verdict.windows, unit), key="windows")
        st.caption("How to read it: each cell is one possible study period. Pick a start year on the bottom axis "
                   "and an end year on the left axis; the colour is the trend over those years. Red means rising, "
                   "blue falling, grey no change. A dot marks periods that pass the noise test on their own. "
                   "A trustworthy trend keeps one colour across the chart. If the colour flips, the answer depends "
                   "on the years you pick.")

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
    st.write(f"Runs the same checks for {len(power.CITIES)} areas ({season.lower()}), then corrects for the fact "
             "that testing many places makes false alarms more likely (Benjamini-Hochberg). A Robust or Abrupt shift "
             "verdict that does not survive the correction becomes Inconclusive here.")
    saved = st.session_state.setdefault("compare", {})
    if (parameter, season) not in saved and st.button("Run all areas"):
        progress = st.progress(0.0)
        rows, errors = scan.judge_places(parameter, season, load=load, progress=progress.progress)
        progress.empty()
        for e in errors:
            st.warning(e)
        saved[(parameter, season)] = rows

    rows = saved.get((parameter, season))
    if rows:
        result = pd.DataFrame(rows).rename(columns={
            "city": "City", "verdict": "Verdict", "slope10": "Change per decade", "lo": "Low", "hi": "High",
            "adj_p": "Adjusted p"})
        result["Plain trend line"] = np.where(result["plain_p"] < stats.ALPHA, "significant", "not significant")
        unit_all = unit_text(result["units"].iloc[0])
        map_col, chart_col = st.columns(2)
        with map_col:
            st.plotly_chart(cities_map(result, unit_all), key="map")
            st.caption("Filled circle = Robust, hollow circle = Inconclusive, diamond = Abrupt shift. "
                       "Colour = change per decade.")
        with chart_col:
            st.plotly_chart(cities_chart(result, unit_all), key="cities")
        shown = result[["City", "Verdict", "Plain trend line", "Change per decade", "Low", "High", "p", "Adjusted p"]].round(3)
        for column in ("p", "Adjusted p"):
            shown[column] = result[column].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
        st.dataframe(shown.sort_values("Change per decade", ascending=False), hide_index=True)
        if parameter == "ALLSKY_SFC_SW_DWN":
            st.caption("Sunlight comes on a 1° grid, so some neighbouring areas (Jeddah and Makkah) share one record. "
                       "Shared records are counted once in the correction.")

        robust = result[result["Verdict"] == stats.ROBUST]
        up = robust[robust["Change per decade"] > 0].sort_values("Change per decade", ascending=False)
        down = robust[robust["Change per decade"] < 0].sort_values("Change per decade")
        if len(up) and len(down):
            st.success(f"Opposite twins: {up.iloc[0]['City']} is rising ({up.iloc[0]['Change per decade']:+.2f} {unit_all} "
                       f"per decade) while {down.iloc[0]['City']} is falling ({down.iloc[0]['Change per decade']:+.2f}). "
                       "Both pass every check.")
        else:
            st.info("No opposite twins here: no two areas have robust trends in opposite directions.")
        plain_sig = result["Plain trend line"] == "significant"
        not_trend = result[plain_sig & (result["Verdict"] == stats.INCONCLUSIVE)]
        jumps = result[plain_sig & (result["Verdict"] == stats.SHIFT)]
        if len(not_trend):
            st.warning("A plain trend line calls these significant, and Mizan does not call them a trend: "
                       + ", ".join(not_trend["City"]) + ". Open each one in the first tab to see why.")
        if len(jumps):
            st.info("A plain trend line calls these a significant trend. Mizan agrees something changed, but finds "
                    "one abrupt jump instead: " + ", ".join(jumps["City"]) + ".")
        st.download_button("Download this table (CSV)", shown.to_csv(index=False),
                           file_name=f"mizan_areas_{parameter}.csv")


# ------------------------------------------------------------------ method

def calibration_table() -> pd.DataFrame | None:
    """The table written by `python calibration.py --csv`, as whole percentages."""
    path = Path(__file__).parent / "results" / "calibration.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)

    def pct(share: float) -> str:
        value = share * 100
        return "0%" if value == 0 else (f"{value:.1f}%" if value < 1 else f"{value:.0f}%")

    out = pd.DataFrame({"Simulated truth": table["case"]})
    for column, name in (("plain_line_significant", "Plain line significant"), ("mizan_robust", "Mizan: Robust"),
                         ("mizan_abrupt_shift", "Mizan: Abrupt shift"),
                         ("mizan_any_significant", "Mizan: any significant change")):
        if column in table:
            out[name] = table[column].map(pct)
    return out


with tab_method:
    st.header("How Mizan decides")
    st.markdown(f"""
Mizan means “balance” in Arabic. A rising line is easy to draw. Mizan weighs the evidence before it calls that line a trend.
It is a **conservative screen**: it raises far fewer false alarms than a spreadsheet trend line, and pays for that by
missing many weak trends.

**The decision rule**: the same checks, in the same order and with the same thresholds, for every place, variable and season.

| Check | Question | Passes when |
|---|---|---|
| Record length | Is there enough data? | At least {stats.MIN_YEARS} complete years |
| Shift check | Is it one jump? | A single step does not fit clearly better than a line (BIC difference of {stats.STEP_BIC_MARGIN:.0f} or less, counting only the effectively independent years), or a significant same-direction trend exists on one side of the step |
| Noise test | Could random variation explain it? | Mann-Kendall p below {stats.ALPHA} after a lag-1 autocorrelation correction, and the 95% range of the slope excludes zero. If the shift check found a jump, the noise is measured around the jump |
| Stress test | Does it survive other start and end years? | At least {stats.AGREEMENT_NEEDED * 100:.0f}% of all periods of {stats.MIN_YEARS}+ years have the same direction as the full record (for records under 30 years: periods of at least 60% of the record) |
| Source check (sunlight only) | Is it just the 2001 switch of data product? | A trend fits better than two flat levels split at 2001, or the trend continues on one side of 2001 |

**Verdicts**

- **Robust:** passes every check. Statistics cannot prove a trend, so we do not use that word. A Robust verdict holds under Mizan's rules; it is not a guarantee.
- **Inconclusive:** fails the noise test, the stress test or the source check. This does not mean nothing is changing. It means this record cannot show it.
- **Abrupt shift:** the change is significant but concentrated in one jump. It may be a real shift or a change in the data record, so it should be compared with another record.

**Why these checks**

- *Size.* The Theil-Sen slope is the median of the slopes between every pair of years. One freak year cannot drag it.
- *Noise test.* Years that resemble their neighbours carry less independent information, so the test is made stricter when we detect that. We use lag-1 (AR(1)) variance inflation, (1 + ρ) / (1 − ρ): the large-sample form of the effective-sample-size correction of Yue & Wang (2004). It is not the full modified Mann-Kendall test of Hamed & Rao (1998).
- *Why the shift check comes first.* A jump makes the leftovers around a straight line look like slow, persistent noise. Measured that way, a clear jump would be called "random variation" for the wrong reason.
- *Stress test.* Periods shorter than {stats.MIN_YEARS} years flip direction easily, so only longer ones count. The periods overlap, so the {stats.AGREEMENT_NEEDED * 100:.0f}% is a stability screen, not a probability. It guards against cherry-picked start years, and it also marks change that only starts late in the record as unstable.
- *Source check.* NASA POWER's sunlight record switches from GEWEX SRB to CERES in 2001. The shift check looks for the single best jump anywhere, so a smaller jump at the switch can hide inside a record and pass as a trend. The source check tests the switch year directly.
- *Many places.* When the {len(power.CITIES)} built-in areas are tested together (the *Compare Saudi areas* tab, and the area count in the first tab), we adjust the p-values with Benjamini-Hochberg, one family per variable and season, counting areas that share a grid cell once. A single place on its own is one test and is not adjusted.

**What Mizan adds to a plain trend line.** A spreadsheet trend line assumes every year is independent and uses every start year you give it.
Mizan reports when the two disagree. Disagreeing is not the same as being right: on real data the truth is unknown,
and some trends Mizan cannot confirm may still be real.

**How well calibrated is Mizan?** We ran Mizan, and an ordinary straight-line fit, on simulated 45-year records where
the truth is known (`calibration.py`, 300 records per case; a rate of 6% is uncertain by about ±1.4 points).
"Any significant change" (Robust or Abrupt shift) is the fair comparison with the plain line.
""")
    table = calibration_table()
    if table is not None:
        st.dataframe(table, hide_index=True)
    st.markdown("""
**Limits**

- Each value is one NASA POWER grid cell: about 55 km for weather (MERRA-2), about 110 km for sunlight. Values describe an area, not a city centre or a mountain top.
- NASA POWER's weather comes from a model (MERRA-2) that absorbed new satellites over time, and its sunlight from satellite products that changed too. A jump can come from the record, not the climate.
- Rainfall verdicts in deserts are unreliable: mostly zeros, rare storms, and weak model rainfall.
- Records start in 1981 (1984 for sunlight). Mizan says nothing about earlier decades.
- Mizan sometimes reads a step as a gradual trend, and misses most weak trends. See the calibration table.
- A robust trend says *that* something changed, not *why*.
""")


st.sidebar.divider()
st.sidebar.caption("Data: NASA POWER (NASA Langley Research Center). Built for the NASA Space Apps Challenge 2026. "
                   "Parts of the code were written with AI assistance (Claude); see the README.")
