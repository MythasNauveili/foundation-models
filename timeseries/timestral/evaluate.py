"""Generate this model's six real-data forecasts; run with its own uv project."""

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
MODEL_ID = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
# filename, horizon, seasonal lag, past-only covariates
DATASETS = {
    "weather": ("strasbourg_daily.csv", 7, 365, ["precipitation_sum", "wind_speed_10m_max"]),
    "bikes": ("bikes_hourly.csv", 24, 168, ["temp", "hum", "windspeed"]),
    "retail": ("retail_monthly.csv", 6, 12, []),
}


def main():
    model = TimestralForecaster.from_pretrained(MODEL_ID, revision=REVISION, device="cpu")
    frames = []
    metadata = {
        "model_id": MODEL_ID, "revision": REVISION,
        "library_version": version("timestral"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "float32", "dataset_sha256": {}, "inference_seconds": {},
    }
    for dataset, (filename, horizon, seasonal_lag, past_columns) in DATASETS.items():
        path = AREA / "datasets" / filename
        metadata["dataset_sha256"][dataset] = sha256(path.read_bytes()).hexdigest()
        data = pd.read_csv(path, parse_dates=["timestamp"])
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

        history = data.iloc[:-horizon].copy()
        # Future inputs contain calendar features only, never targets or observed weather.
        future_known = data.iloc[-horizon:][["timestamp", *known_columns]].copy()
        actual = data.target.iloc[-horizon:].to_numpy()
        baseline = history.target.iloc[-seasonal_lag:-seasonal_lag + horizon].to_numpy().copy()
        for step in range(horizon):
            if pd.isna(baseline[step]):
                baseline[step] = history.target.iloc[step - seasonal_lag::-seasonal_lag].dropna().iloc[0]
        assert np.isfinite(actual).all() and np.isfinite(baseline).all()

        metadata["inference_seconds"][dataset] = {}
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
            quantiles = np.stack([output.quantile(q)[0, 0].numpy() for q in (0.1, 0.5, 0.9)], axis=-1)
            assert quantiles.shape == (horizon, 3) and np.isfinite(quantiles).all()
            assert (np.diff(quantiles, axis=1) >= 0).all()
            metadata["inference_seconds"][dataset][mode] = elapsed
            frames.append(pd.DataFrame({
                "dataset": dataset, "mode": mode, "timestamp": future_known.timestamp,
                "actual": actual, "baseline": baseline,
                "p10": quantiles[:, 0], "p50": quantiles[:, 1], "p90": quantiles[:, 2],
            }))
            print(f"{dataset} / {mode}: MAE={np.abs(actual - quantiles[:, 1]).mean():.3f}", flush=True)

    results = AREA / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / "timestral.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    metadata["predictions_sha256"] = sha256(csv_path.read_bytes()).hexdigest()
    (results / "timestral.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
