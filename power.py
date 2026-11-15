"""
Mizan data access: monthly values from the NASA POWER API.

NASA POWER (https://power.larc.nasa.gov/) serves weather and solar values for any
point on Earth. No login or API key is needed. The values come from different
NASA products depending on the variable (see SOURCES below):

  - Weather (temperature, rainfall, humidity, wind): NASA GMAO's MERRA-2
    reanalysis model, from 1981.
  - Sunlight reaching the surface (ALLSKY_SFC_SW_DWN): NASA GEWEX SRB
    Release 4.0-IP from 1984 to 2000, then CERES SYN1deg from 2001.
    The change of product in 2001 can create a step in a sunlight record.

Source list: https://power.larc.nasa.gov/docs/methodology/data/sources_fluxes/

AI assistance: this file was drafted with Claude (Anthropic) during the hackathon
and reviewed and tested by the team. See README.md.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
FIRST_YEAR = 1981
TIMEOUT = 30          # seconds per request
RETRY_WAIT = 2        # seconds before the one retry after a timeout or a busy server
DATA_DIR = Path(__file__).parent / "data"   # saved copies, so the app works if the API is down

# What the user can pick. Labels and units are replaced by NASA's own wording
# when the API returns it.
VARIABLES = {
    "T2M": "Air temperature at 2 m",
    "T2M_MAX": "Maximum air temperature at 2 m",
    "T2M_MIN": "Minimum air temperature at 2 m",
    "PRECTOTCORR": "Precipitation",
    "RH2M": "Relative humidity at 2 m",
    "WS10M": "Wind speed at 10 m",
    "ALLSKY_SFC_SW_DWN": "Sunlight reaching the surface",
}

# Which NASA product each variable comes from, as listed in NASA POWER's data
# sources. Shown in the app's "Data source" panel and in the README.
MERRA2 = "NASA GMAO MERRA-2 reanalysis model (1981 onwards)"
SOLAR = ("NASA GEWEX SRB Release 4.0-IP (1984-2000), then CERES SYN1deg (2001 onwards). "
         "The record switches product in 2001, so a step near that year may come from the switch.")
SOURCES = {
    "T2M": MERRA2,
    "T2M_MAX": MERRA2,
    "T2M_MIN": MERRA2,
    "PRECTOTCORR": MERRA2,
    "RH2M": MERRA2,
    "WS10M": MERRA2,
    "ALLSKY_SFC_SW_DWN": SOLAR,
}
SOURCES_URL = "https://power.larc.nasa.gov/docs/methodology/data/sources_fluxes/"


# Years at which a record switches data product. A jump at that year may be an
# artifact of the switch, so stats.judge(known_break=...) checks it explicitly.
SOURCE_CHANGE = {"ALLSKY_SFC_SW_DWN": 2001}
SOURCE_CHANGE_NAME = {"ALLSKY_SFC_SW_DWN": "GEWEX SRB to CERES"}

# Grid size behind each value. MERRA-2 is 0.5 x 0.625 degrees; POWER's solar
# values are on a 1 x 1 degree grid, so nearby places can share one record
# (Jeddah and Makkah have the same sunlight series).
MERRA2_GRID = "one MERRA-2 grid cell, 0.5° × 0.625° (roughly 55 km across; narrower east-west far from the equator)"
SOLAR_GRID = ("one solar grid cell, 1° × 1° (roughly 110 km across; narrower east-west far from the equator). "
              "Nearby places can share a cell")
RESOLUTION = {k: SOLAR_GRID if k == "ALLSKY_SFC_SW_DWN" else MERRA2_GRID for k in VARIABLES}

# One line each, for readers who do not know the variable names.
DESCRIPTIONS = {
    "T2M": "Average air temperature 2 metres above the ground, day and night.",
    "T2M_MAX": "The hottest temperature of each day, averaged over the month.",
    "T2M_MIN": "The coolest temperature of each day (usually at night), averaged over the month.",
    "PRECTOTCORR": "Rain and snow reaching the ground, in millimetres per day, bias-corrected by NASA.",
    "RH2M": "How close the air is to saturation with water vapour, in percent.",
    "WS10M": "Average wind speed 10 metres above the ground.",
    "ALLSKY_SFC_SW_DWN": "Sunlight energy reaching the ground per day, including the effect of clouds and dust.",
}


def source_of(parameter: str) -> str:
    """The NASA product behind one POWER parameter."""
    return SOURCES.get(parameter, "NASA POWER (see " + SOURCES_URL + ")")


SEASONS = {
    "Whole year": list(range(1, 13)),
    "Winter (Dec-Feb)": [12, 1, 2],
    "Spring (Mar-May)": [3, 4, 5],
    "Summer (Jun-Aug)": [6, 7, 8],
    "Autumn (Sep-Nov)": [9, 10, 11],
}

# Approximate city-centre coordinates. POWER's grid cells are about 50 km wide,
# so each value describes the area around the city, not one street.
CITIES = {
    "Jeddah": (21.54, 39.17),
    "Makkah": (21.39, 39.86),
    "Madinah": (24.47, 39.61),
    "Taif": (21.27, 40.42),
    "Yanbu": (24.09, 38.06),
    "Riyadh": (24.71, 46.68),
    "Buraydah": (26.33, 43.97),
    "Hail": (27.52, 41.70),
    "Dammam": (26.43, 50.10),
    "Al-Ahsa": (25.38, 49.59),
    "Tabuk": (28.38, 36.57),
    "Arar": (30.98, 41.02),
    "Abha": (18.22, 42.51),
    "Jazan": (16.89, 42.57),
    "Najran": (17.49, 44.13),
}


def build_url(lat: float, lon: float, parameter: str, start: int, end: int) -> str:
    return (f"{API}?parameters={parameter}&community=AG&latitude={lat}&longitude={lon}"
            f"&start={start}&end={end}&format=JSON")


def parse(payload: dict, parameter: str) -> tuple[pd.DataFrame, dict]:
    """Turn a POWER response into a table of year, month, value. Missing months are dropped."""
    try:
        raw = payload["properties"]["parameter"][parameter]
    except (KeyError, TypeError) as err:
        raise ValueError(f"NASA POWER did not return '{parameter}'. Response keys: {list(payload)[:6]}") from err

    fill = payload.get("header", {}).get("fill_value", -999.0)
    rows = []
    for key, value in raw.items():
        year, month = int(key[:4]), int(key[4:6])
        if not 1 <= month <= 12:          # POWER adds a 13th entry holding the annual value
            continue
        if value is None or value == fill or value <= -990:
            continue
        rows.append((year, month, float(value)))
    table = pd.DataFrame(rows, columns=["year", "month", "value"]).sort_values(["year", "month"])

    info = payload.get("parameters", {}).get(parameter, {})
    meta = {"units": info.get("units", ""), "longname": info.get("longname", "")}
    return table.reset_index(drop=True), meta


def _copy_paths(lat: float, lon: float, parameter: str) -> tuple[Path, Path]:
    stem = f"{parameter}_{lat:.2f}_{lon:.2f}"
    return DATA_DIR / f"{stem}.csv", DATA_DIR / f"{stem}.json"


def save_copy(lat: float, lon: float, parameter: str, table: pd.DataFrame, meta: dict) -> None:
    csv_path, json_path = _copy_paths(lat, lon, parameter)
    DATA_DIR.mkdir(exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2))


def _get(url: str) -> requests.Response:
    """One request, retried once after a short wait if it times out or the server is busy."""
    for attempt in (1, 2):
        try:
            response = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as err:
            if attempt == 1:
                time.sleep(RETRY_WAIT)
                continue
            raise RuntimeError(f"Could not reach NASA POWER. Check your internet connection. ({err})") from err
        if (response.status_code == 429 or response.status_code >= 500) and attempt == 1:
            time.sleep(RETRY_WAIT)
            continue
        return response
    return response


def fetch(lat: float, lon: float, parameter: str, save: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Monthly record for one point. Uses the saved copy in data/ if there is one,
    otherwise downloads from NASA POWER.

    save=True keeps a copy in data/ (prefetch.py uses it for the built-in areas).
    The app never saves, so custom points do not pile up on the server's disk;
    Streamlit's own cache keeps them for a day instead.
    """
    csv_path, json_path = _copy_paths(lat, lon, parameter)
    if csv_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        meta["saved_copy"] = True
        meta["source"] = source_of(parameter)
        return pd.read_csv(csv_path), meta

    last_error = None
    this_year = dt.date.today().year
    for end in (this_year - 1, this_year - 2):   # the newest completed year may not be published yet
        url = build_url(lat, lon, parameter, FIRST_YEAR, end)
        response = _get(url)
        if response.ok:
            table, meta = parse(response.json(), parameter)
            meta.update(url=url, accessed=dt.date.today().isoformat(), source=source_of(parameter))
            if save:
                try:
                    save_copy(lat, lon, parameter, table, meta)
                except OSError:
                    pass   # read-only disk: carry on without saving
            meta["saved_copy"] = False
            return table, meta
        last_error = f"NASA POWER answered {response.status_code}: {response.text[:300]}"
        if response.status_code == 429 or response.status_code >= 500:
            break   # busy or down: an earlier end year will not help
    raise RuntimeError(last_error)


def yearly(table: pd.DataFrame, months: list[int]):
    """
    One value per year: the mean of the chosen months.
    A year is kept only if every chosen month is present.
    For winter, December counts toward the winter of the following year.
    """
    part = table[table["month"].isin(months)].copy()
    if 12 in months and 1 in months and len(months) < 12:
        part.loc[part["month"] >= 10, "year"] += 1
    grouped = part.groupby("year")["value"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] == len(months)]
    return grouped.index.to_numpy(dtype=float), grouped["mean"].to_numpy(dtype=float)
