"""Declare the question before running a model. All timestamps mark observation bins."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
REGIONS = {
    "44": "Grand Est",
    "53": "Bretagne",
    "11": "Île-de-France",
    "76": "Occitanie",
    "32": "Hauts-de-France",
    "93": "Provence-Alpes-Côte d'Azur",
}
TRACKS = {
    "target_only": "Target only",
    "calendar": "Calendar",
    "past_observed": "Past observed weather",
    "past_observed_calendar": "Past observed weather + calendar",
}
INPUT_VERSION = 1  # Increment when loading, window, or feature semantics change.
SCHEDULE_COLUMNS = ["task", "entity", "forecast_start", "split"]
TEACHING_TASKS = ("weather-daily-7-v1", "bikes-2hourly-12-v1", "retail-monthly-6-v1")


def require(condition, message):
    """Input validation must also run under python -O."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DatasetSpec:
    file: str
    title: str
    unit: str
    utc: bool = False
    calendar_timezone: str | None = None
    entity_labels: tuple = ()
    missing_policy: str = "native"
    past_columns: tuple = ()
    calendar_columns: tuple = ()
    card: str = ""


@dataclass(frozen=True)
class TaskSpec:
    id: str
    dataset: str
    history_length: int
    horizon: int
    frequency: str
    seasonal_period: int
    tracks: tuple
    baselines: tuple
    reference: str
    primary_metric: str = "MAE"
    default_split: str = "exploratory"
    allowed_splits: tuple = ("exploratory",)


def persistence(target, horizon):
    """Repeat the last observed value, leaving unknown history undefined."""
    observed = np.asarray(target, dtype=float)
    observed = observed[np.isfinite(observed)]
    return np.full(horizon, observed[-1] if len(observed) else np.nan)


def _seasonal_references(target, horizon, lag, seasons=None):
    """Visit matching historical phases, newest first; never use future values."""
    target = np.asarray(target, dtype=float)
    require(lag > 0, "seasonal lag must be positive")
    for step in range(horizon):
        position = len(target) - lag + step % lag
        indices = np.arange(position, -1, -lag, dtype=int)[:seasons]
        values = target[indices]
        yield values[np.isfinite(values)]


def seasonal_naive(target, horizon, *, lag):
    """Use the latest finite observation at each seasonal phase."""
    return np.array([v[0] if len(v) else np.nan for v in _seasonal_references(target, horizon, lag)])


def seasonal_average(target, horizon, *, lag, seasons=4):
    """Average available values among exactly the requested seasonal references."""
    require(seasons > 0, "number of seasons must be positive")
    return np.array([v.mean() if len(v) else np.nan
                     for v in _seasonal_references(target, horizon, lag, seasons)])


DATASETS = {
    "weather": DatasetSpec(
        "strasbourg_daily.csv",
        "Strasbourg weather · daily",
        "°C",
        past_columns=("precipitation_sum", "wind_speed_10m_max"),
        calendar_columns=("day_of_year",),
        card="datasets/weather.md",
    ),
    "bikes": DatasetSpec(
        "bikes_2hourly.csv",
        "Bike demand · every two hours",
        "rentals / 2 hours",
        past_columns=("temp", "hum", "windspeed"),
        calendar_columns=("hour", "weekday"),
        card="datasets/bikes.md",
    ),
    "retail": DatasetSpec(
        "retail_monthly.csv",
        "Retail revenue · monthly",
        "million USD",
        calendar_columns=("month",),
        card="datasets/retail.md",
    ),
    "rte": DatasetSpec(
        "rte_hourly.csv",
        "RTE regional electricity · hourly",
        "MW",
        utc=True,
        calendar_timezone="Europe/Paris",
        entity_labels=tuple(REGIONS.items()),
        missing_policy="complete",
        calendar_columns=("hour", "weekday"),
        card="datasets/rte.md",
    ),
}
_FOUR = ("target_only", "calendar", "past_observed", "past_observed_calendar")
_TWO = ("target_only", "calendar")
TASKS = {
    t.id: t
    for t in [
        TaskSpec(
            id="weather-daily-7-v1",
            dataset="weather",
            history_length=730,
            horizon=7,
            frequency="D",
            seasonal_period=365,
            tracks=_FOUR,
            baselines=(("Persistence", persistence), ("Seasonal naïve", partial(seasonal_naive, lag=365))),
            reference="Seasonal naïve",
        ),
        TaskSpec(
            id="bikes-2hourly-12-v1",
            dataset="bikes",
            history_length=4380,
            horizon=12,
            frequency="2h",
            seasonal_period=84,
            tracks=_FOUR,
            baselines=(
                ("Persistence", persistence), ("Seasonal naïve", partial(seasonal_naive, lag=84)),
                ("Four-week average", partial(seasonal_average, lag=84, seasons=4)),
            ),
            reference="Seasonal naïve",
        ),
        TaskSpec(
            id="retail-monthly-6-v1",
            dataset="retail",
            history_length=120,
            horizon=6,
            frequency="MS",
            seasonal_period=12,
            tracks=_TWO,
            baselines=(("Persistence", persistence), ("Seasonal naïve", partial(seasonal_naive, lag=12))),
            reference="Seasonal naïve",
        ),
        TaskSpec(
            id="rte-hourly-24-v1",
            dataset="rte",
            history_length=1344,
            horizon=24,
            frequency="h",
            seasonal_period=168,
            tracks=_TWO,
            baselines=(
                ("Persistence", persistence), ("Daily naïve", partial(seasonal_naive, lag=24)),
                ("Weekly naïve", partial(seasonal_naive, lag=168)),
                ("Four-week average", partial(seasonal_average, lag=168, seasons=4)),
            ),
            reference="Weekly naïve",
            primary_metric="MASE",
            default_split="development",
            allowed_splits=("development",),
        ),
    ]
}


def calendar_features(timestamps, columns, timezone=None):
    """Known calendar values only; numeric encoding is deliberate and versioned."""
    clock = timestamps.dt.tz_convert(timezone) if timezone else timestamps
    accessors = {
        "day_of_year": "dayofyear",
        "hour": "hour",
        "weekday": "dayofweek",
        "month": "month",
    }
    return {name: getattr(clock.dt, accessors[name]).astype(float) for name in columns}


def load_data(task, area=AREA):
    spec = DATASETS[task.dataset]
    data = pd.read_csv(area / "datasets" / spec.file, dtype={"entity": str})
    data["entity"] = data.get("entity", "single")
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=spec.utc)
    require(not data.entity.isna().any(), "missing entity")
    require(not data.duplicated(["entity", "timestamp"]).any(), "duplicate entity/timestamp")
    for _, group in data.groupby("entity", sort=False):
        expected = pd.date_range(
            group.timestamp.iloc[0], group.timestamp.iloc[-1], freq=task.frequency
        )
        require(
            pd.DatetimeIndex(group.timestamp).equals(expected), "unordered or irregular timestamps"
        )
    for name, values in calendar_features(
        data.timestamp, spec.calendar_columns, spec.calendar_timezone
    ).items():
        data[name] = values
    return data


def normalize_cycles(frame, task):
    cycles = frame.loc[frame.task == task.id, SCHEDULE_COLUMNS].copy()
    cycles["entity"] = cycles.entity.astype(str)
    cycles["forecast_start"] = pd.to_datetime(cycles.forecast_start, utc=DATASETS[task.dataset].utc)
    require(not cycles.duplicated(["entity", "forecast_start"]).any(), "duplicate forecast origin")
    require(cycles.split.isin(["exploratory", "development", "evaluation"]).all(), "invalid split")
    return cycles.sort_values(["entity", "forecast_start"]).reset_index(drop=True)


def load_cycles(task, area=AREA):
    return normalize_cycles(
        pd.read_csv(area / "datasets" / "cycles.csv", dtype={"entity": str}), task
    )


def select_cycles(task, split=None, smoke=False, area=AREA):
    require(
        task.history_length > 0 and task.horizon > 0 and task.seasonal_period > 0,
        "task lengths must be positive",
    )
    require(
        bool(task.tracks) and len(set(task.tracks)) == len(task.tracks), "empty or duplicate tracks"
    )
    for track in task.tracks:
        input_columns(task, track)
    split = split or task.default_split
    require(split in task.allowed_splits, f"Reserved or unsupported split: {split}")
    cycles = load_cycles(task, area)
    cycles = cycles.loc[cycles.split == split].reset_index(drop=True)
    require(len(cycles) > 0, "no eligible windows")
    return cycles.head(1).copy() if smoke else cycles


def forecast_window(data, cycle, task):
    """Slice one origin from the regular grid already checked by load_data."""
    series = data.loc[data.entity == cycle.entity].set_index("timestamp", drop=False)
    position = series.index.get_loc(cycle.forecast_start)
    require(position >= task.history_length, "insufficient history")
    history = series.iloc[position - task.history_length : position].copy()
    future = series.iloc[position : position + task.horizon].copy()
    require(len(future) == task.horizon, "incomplete horizon")
    require(np.isfinite(future.target).all(), "missing evaluation targets")
    if DATASETS[task.dataset].missing_policy == "complete":
        require(np.isfinite(history.target).all(), "missing complete-history targets")
    return history, future


def baselines(history, task):
    return {name: forecast(history.target, task.horizon) for name, forecast in task.baselines}


def input_columns(task, track):
    require(track in task.tracks and track in TRACKS, "unsupported task track")
    spec = DATASETS[task.dataset]
    past = track in ("past_observed", "past_observed_calendar")
    known = track in ("calendar", "past_observed_calendar")
    require(not past or bool(spec.past_columns), "empty past-observed track")
    require(not known or bool(spec.calendar_columns), "empty calendar track")
    return spec.past_columns if past else (), spec.calendar_columns if known else ()


@dataclass(frozen=True)
class ForecastInput:
    target: np.ndarray
    past_covariates: dict
    known_history: dict
    known_future: dict
    horizon: int


def build_input(history, future_timestamps, task, track):
    spec = DATASETS[task.dataset]
    past, known = input_columns(task, track)

    future_clock = pd.Series(future_timestamps)
    future_values = calendar_features(future_clock, known, spec.calendar_timezone)
    array = lambda values: np.asarray(values, dtype=np.float32).copy()
    return ForecastInput(
        array(history.target),
        {c: array(history[c]) for c in past},
        {c: array(history[c]) for c in known},
        {c: array(future_values[c]) for c in known},
        task.horizon,
    )


def task_contract(task):
    """Prediction compatibility depends on inputs, not labels or Python source text."""
    spec = DATASETS[task.dataset]
    return {
        "input_version": INPUT_VERSION,
        "task": task.id,
        "dataset": task.dataset,
        "history_length": task.history_length,
        "horizon": task.horizon,
        "frequency": task.frequency,
        "utc": spec.utc,
        "calendar_timezone": spec.calendar_timezone,
        "missing_policy": spec.missing_policy,
        "tracks": {track: [list(cols) for cols in input_columns(task, track)] for track in task.tracks},
    }


def task_label(task):
    return f"{DATASETS[task.dataset].title} · {task.horizon} steps ({task.id})"
