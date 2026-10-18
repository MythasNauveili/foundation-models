"""Recreate the three teaching CSVs and their declared task schedules."""

from datetime import date
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from foundation_timeseries.tasks import SCHEDULE_COLUMNS, TASKS


DESTINATION = Path(__file__).resolve().parent


def download(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def save(frame, name, frequency):
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq=frequency))
    frame.rename_axis("timestamp").to_csv(DESTINATION / name, float_format="%.6f")
    print(f"{name}: {len(frame)} rows, {frame.index[0]} to {frame.index[-1]}")


def weather():
    query = urlencode(
        {
            "latitude": 48.58392,
            "longitude": 7.74553,
            "start_date": "2023-01-01",
            "end_date": "2025-12-31",
            "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
            "timezone": "Europe/Paris",
            "models": "era5",
        }
    )
    data = json.loads(download(f"https://archive-api.open-meteo.com/v1/archive?{query}"))
    frame = pd.DataFrame(data["daily"]).set_index("time")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.rename(columns={"temperature_2m_max": "target"})
    assert len(frame) == 1096 and frame.notna().all().all()
    save(frame, "strasbourg_daily.csv", "D")
    return frame


def bikes():
    archive = ZipFile(
        BytesIO(download("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"))
    )
    with archive.open("hour.csv") as stream:
        frame = pd.read_csv(stream)
    frame.index = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    frame = frame[["cnt", "temp", "hum", "windspeed"]].rename(columns={"cnt": "target"})
    # Preserve source local clock labels; absent hours are unknown, not zero rentals.
    frame = frame.sort_index().asfreq("h")
    two_hourly = frame.resample("2h").mean().where(frame.resample("2h").count().eq(2))
    two_hourly["target"] = frame.target.resample("2h").sum(min_count=2)
    assert len(two_hourly) == 8772 and two_hourly.target.isna().sum() == 121
    save(two_hourly, "bikes_2hourly.csv", "2h")
    return two_hourly


def retail():
    frame = pd.read_csv(
        BytesIO(
            download(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MRTSSM44000USN&cosd=2014-01-01&coed=2025-12-01"
            )
        ),
        index_col=0,
        parse_dates=True,
    )
    frame.columns = ["target"]
    frame = frame.loc["2014-01-01":"2025-12-01"].apply(pd.to_numeric)
    assert len(frame) == 144 and frame.notna().all().all()
    save(frame, "retail_monthly.csv", "MS")
    return frame


def save_cycles(frames, starts=None):
    if starts is None:
        starts = {
            "weather-daily-7-v1": [
                end - pd.Timedelta(days=6)
                for end in pd.date_range("2025-01-31", "2025-12-31", freq="ME")
            ],
            "retail-monthly-6-v1": pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]),
        }
        complete_days = frames["bikes"].loc["2012", "target"].notna().resample("D").sum().eq(12)
        available = complete_days[complete_days].index
        starts["bikes-2hourly-12-v1"] = [available[available.month == month][-1] for month in range(1, 13)]
    cycles = []
    for task_id, origins in starts.items():
        task = TASKS[task_id]
        frame = frames[task.dataset]
        for start in origins:
            position = frame.index.get_loc(start)
            history = frame.iloc[position - task.history_length : position]
            future = frame.iloc[position : position + task.horizon]
            assert len(history) == task.history_length and len(future) == task.horizon
            assert future.target.notna().all(), f"Incomplete {task.id} evaluation at {start}"
            cycles.append(dict(task=task.id, entity="single", forecast_start=start, split=task.default_split))
    schedule = pd.DataFrame(cycles, columns=SCHEDULE_COLUMNS)
    path = DESTINATION / "cycles.csv"
    if path.exists():
        previous = pd.read_csv(path, dtype={"entity": str})
        schedule = pd.concat(
            [schedule, previous[~previous.task.isin(starts)]], ignore_index=True
        )
    schedule.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    (DESTINATION / "bikes_hourly.csv").unlink(missing_ok=True)
    print(f"cycles.csv: {len(cycles)} fixed forecast cycles")


if __name__ == "__main__":
    frames = {"weather": weather(), "bikes": bikes(), "retail": retail()}
    save_cycles(frames)
    print(f"Retrieved {date.today().isoformat()}; review CSV changes before committing a refresh.")
