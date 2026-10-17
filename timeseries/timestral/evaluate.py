"""Generate seasonal forecast cycles per Timestral size, using its shared uv project."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from timestral import TimestralForecaster

AREA = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "180698bbf5a50dda3516aa6419798437fb03a430",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
# filename, past-only covariates
DATASETS = {
    "weather": ("strasbourg_daily.csv", ["precipitation_sum", "wind_speed_10m_max"]),
    "bikes": ("bikes_2hourly.csv", ["temp", "hum", "windspeed"]),
    "retail": ("retail_monthly.csv", []),
}


def evaluate(size):
    model_id = f"mistralai/Timestral-{size}"
    revision = CHECKPOINTS[size]
    model = TimestralForecaster.from_pretrained(model_id, revision=revision, device="cpu")
    schedule_path = AREA / "datasets" / "cycles.csv"
    schedule = pd.read_csv(schedule_path, parse_dates=["forecast_start"])
    frames = []
    metadata = {
        "schema_version": 2,
        "cycles_sha256": sha256(schedule_path.read_bytes()).hexdigest(),
        "cycles": pd.read_csv(schedule_path).to_dict("records"),
        "model_id": model_id, "revision": revision,
        "library_version": version("timestral"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "float32", "dataset_sha256": {}, "inference_seconds": {},
    }
    for dataset, (filename, past_columns) in DATASETS.items():
        path = AREA / "datasets" / filename
        metadata["dataset_sha256"][dataset] = sha256(path.read_bytes()).hexdigest()
        data = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp", drop=False)
        if dataset == "weather":
            data["day_of_year"] = data.timestamp.dt.dayofyear.astype(float)
            known_columns = ["day_of_year"]
        elif dataset == "bikes":
            data["hour"] = data.timestamp.dt.hour.astype(float)
            data["weekday"] = data.timestamp.dt.dayofweek.astype(float)
            known_columns = ["hour", "weekday"]
        else:
            data["month"] = data.timestamp.dt.month.astype(float)
            known_columns = ["month"]

        metadata["inference_seconds"][dataset] = {"Target only": {}, "With covariates": {}}
        for cycle in schedule[schedule.dataset == dataset].itertuples(index=False):
            position = data.index.get_loc(cycle.forecast_start)
            horizon, seasonal_lag = cycle.horizon, cycle.seasonal_lag
            history = data.iloc[position - cycle.history_length:position].copy()
            assert len(history) == cycle.history_length
            assert cycle.history_length <= model.max_context, "History exceeds the checkpoint context limit"
            # Future inputs contain calendar features only, never targets or observed weather.
            future_known = data.iloc[position:position + horizon][["timestamp", *known_columns]].copy()
            actual = data.target.iloc[position:position + horizon].to_numpy()
            baseline = history.target.iloc[-seasonal_lag:-seasonal_lag + horizon].to_numpy().copy()
            for step in range(horizon):
                if pd.isna(baseline[step]):
                    earlier = history.target.iloc[step - seasonal_lag::-seasonal_lag].dropna()
                    assert not earlier.empty, f"No historical seasonal baseline: {dataset}, {cycle.forecast_start}, step {step + 1}"
                    baseline[step] = earlier.iloc[0]
            assert np.isfinite(actual).all() and np.isfinite(baseline).all()

            for with_covariates in (False, True):
                mode = "With covariates" if with_covariates else "Target only"
                past = None
                known = None
                if with_covariates:
                    if past_columns:
                        past = history[past_columns].to_numpy(dtype=np.float32).T[None, ...]
                    known = np.concatenate([
                        history[known_columns].to_numpy(dtype=np.float32),
                        future_known[known_columns].to_numpy(dtype=np.float32),
                    ]).T[None, ...]
                started = perf_counter()
                output = model.forecast(
                    context=history.target.to_numpy(dtype=np.float32), prediction_length=horizon,
                    past_covariates=past, known_covariates=known,
                )
                elapsed = perf_counter() - started
                # Preserve the native float32 values when CSV formatting writes decimal numbers.
                quantiles = np.stack([output.quantile(q)[0, 0].numpy() for q in (0.1, 0.5, 0.9)], axis=-1).astype(float)
                assert quantiles.shape == (horizon, 3) and np.isfinite(quantiles).all()
                assert (np.diff(quantiles, axis=1) >= 0).all()
                metadata["inference_seconds"][dataset][mode][str(cycle.forecast_start)] = elapsed
                frames.append(pd.DataFrame({
                    "dataset": dataset, "mode": mode, "forecast_start": cycle.forecast_start,
                    "timestamp": future_known.timestamp, "step": np.arange(1, horizon + 1),
                    "actual": actual, "baseline": baseline,
                    "p10": quantiles[:, 0], "p50": quantiles[:, 1], "p90": quantiles[:, 2],
                }))
                print(f"Timestral-{size} / {dataset} / {cycle.forecast_start:%Y-%m-%d} / {mode}: MAE={np.abs(actual - quantiles[:, 1]).mean():.3f}", flush=True)

    results = AREA / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / f"timestral-{size.lower()}.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    metadata["predictions_sha256"] = sha256(csv_path.read_bytes()).hexdigest()
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["all", *CHECKPOINTS], default="all", help="Checkpoint size (default: all)")
    args = parser.parse_args()
    for size in CHECKPOINTS if args.size == "all" else [args.size]:
        evaluate(size)
