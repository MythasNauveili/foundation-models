"""Recreate the three small reference CSVs; run through either model's uv project."""

from datetime import date
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


DESTINATION = Path(__file__).resolve().parent


def download(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def save(frame, name, frequency, horizon):
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq=frequency))
    assert frame.iloc[-horizon:].notna().all().all(), "Incomplete evaluation window"
    frame.rename_axis("timestamp").to_csv(DESTINATION / name, float_format="%.6f")
    print(f"{name}: {len(frame)} rows, {frame.index[0]} to {frame.index[-1]}")


def weather():
    query = urlencode({
        "latitude": 48.58392,
        "longitude": 7.74553,
        "start_date": "2023-12-26",
        "end_date": "2025-12-31",
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
        "timezone": "Europe/Paris",
        "models": "era5",
    })
    data = json.loads(download(f"https://archive-api.open-meteo.com/v1/archive?{query}"))
    frame = pd.DataFrame(data["daily"]).set_index("time")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.rename(columns={"temperature_2m_max": "target"})
    assert len(frame) == 737 and frame.notna().all().all()
    save(frame, "strasbourg_daily.csv", "D", 7)


def bikes():
    archive = ZipFile(BytesIO(download(
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    )))
    with archive.open("hour.csv") as stream:
        frame = pd.read_csv(stream)
    frame.index = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    frame = frame[["cnt", "temp", "hum", "windspeed"]].rename(columns={"cnt": "target"})
    # Preserve source local clock labels; absent hours are unknown, not zero rentals.
    frame = frame.sort_index().asfreq("h")
    complete_days = frame.loc["2012"].notna().all(axis=1).resample("D").sum()
    final_day = complete_days[complete_days.eq(24)].index[-1]
    frame = frame.loc[final_day - pd.Timedelta(hours=672):final_day + pd.Timedelta(hours=23)]
    assert len(frame) == 696
    save(frame, "bikes_hourly.csv", "h", 24)


def retail():
    frame = pd.read_csv(BytesIO(download(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MRTSSM44000USN&cosd=2015-07-01&coed=2025-12-01"
    )), index_col=0, parse_dates=True)
    frame.columns = ["target"]
    frame = frame.loc["2015-07-01":"2025-12-01"].apply(pd.to_numeric)
    assert len(frame) == 126 and frame.notna().all().all()
    save(frame, "retail_monthly.csv", "MS", 6)


if __name__ == "__main__":
    weather()
    bikes()
    retail()
    print(f"Retrieved {date.today().isoformat()}; review CSV changes before committing a refresh.")
