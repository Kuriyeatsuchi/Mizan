"""
Mizan data access: monthly values from the NASA POWER API.

NASA POWER (https://power.larc.nasa.gov/) serves weather and solar values for any
point on Earth. Its meteorology comes from NASA's MERRA-2 model and starts in 1981.
No login or API key is needed.

AI assistance: this file was drafted with Claude (Anthropic) during the hackathon
and reviewed and tested by the team. See README.md.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

API = "https://power.larc.nasa.gov/api/temporal/monthly/point"
FIRST_YEAR = 1981
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


def fetch(lat: float, lon: float, parameter: str) -> tuple[pd.DataFrame, dict]:
    """
    Monthly record for one point. Uses the saved copy in data/ if there is one,
    otherwise downloads from NASA POWER and saves a copy for next time.
    """
    csv_path, json_path = _copy_paths(lat, lon, parameter)
    if csv_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        meta["saved_copy"] = True
        return pd.read_csv(csv_path), meta

    last_error = None
    this_year = dt.date.today().year
    for end in (this_year - 1, this_year - 2):   # the newest completed year may not be published yet
        url = build_url(lat, lon, parameter, FIRST_YEAR, end)
        try:
            response = requests.get(url, timeout=60)
        except requests.RequestException as err:
            raise RuntimeError(f"Could not reach NASA POWER. Check your internet connection. ({err})") from err
        if response.ok:
            table, meta = parse(response.json(), parameter)
            meta.update(url=url, accessed=dt.date.today().isoformat())
            try:
                save_copy(lat, lon, parameter, table, meta)
            except OSError:
                pass   # read-only disk: carry on without saving
            meta["saved_copy"] = False
            return table, meta
        last_error = f"NASA POWER answered {response.status_code}: {response.text[:300]}"
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
