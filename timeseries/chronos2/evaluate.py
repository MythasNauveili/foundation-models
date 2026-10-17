"""Generate seasonal forecast cycles using the Chronos-2 uv project."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

AREA = Path(__file__).resolve().parents[1]
MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
# filename, past-only covariates
DATASETS = {
    "weather": ("strasbourg_daily.csv", ["precipitation_sum", "wind_speed_10m_max"]),
    "bikes": ("bikes_2hourly.csv", ["temp", "hum", "windspeed"]),
    "retail": ("retail_monthly.csv", []),
}


def main():
    model = Chronos2Pipeline.from_pretrained(
        MODEL_ID, revision=REVISION, device_map="cpu", dtype=torch.float32,
    )
    schedule_path = AREA / "datasets" / "cycles.csv"
    schedule = pd.read_csv(schedule_path, parse_dates=["forecast_start"])
    frames = []
    metadata = {
        "schema_version": 2,
        "cycles_sha256": sha256(schedule_path.read_bytes()).hexdigest(),
        "cycles": pd.read_csv(schedule_path).to_dict("records"),
        "model_id": MODEL_ID, "revision": REVISION,
        "library_version": version("chronos-forecasting"),
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
            assert cycle.history_length <= model.model_context_length, "History exceeds the checkpoint context limit"
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
                inputs = {"target": history.target.to_numpy(dtype=np.float32)}
                if with_covariates:
                    inputs["past_covariates"] = {
                        column: history[column].to_numpy(dtype=np.float32)
                        for column in [*past_columns, *known_columns]
                    }
                    inputs["future_covariates"] = {
                        column: future_known[column].to_numpy(dtype=np.float32) for column in known_columns
                    }
                started = perf_counter()
                output, _ = model.predict_quantiles(
                    inputs=[inputs], prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
                    context_length=cycle.history_length,
                )
                elapsed = perf_counter() - started
                quantiles = output[0][0].numpy().astype(float)
                assert quantiles.shape == (horizon, 3) and np.isfinite(quantiles).all()
                assert (np.diff(quantiles, axis=1) >= 0).all()
                metadata["inference_seconds"][dataset][mode][str(cycle.forecast_start)] = elapsed
                frames.append(pd.DataFrame({
                    "dataset": dataset, "mode": mode, "forecast_start": cycle.forecast_start,
                    "timestamp": future_known.timestamp, "step": np.arange(1, horizon + 1),
                    "actual": actual, "baseline": baseline,
                    "p10": quantiles[:, 0], "p50": quantiles[:, 1], "p90": quantiles[:, 2],
                }))
                print(f"{dataset} / {cycle.forecast_start:%Y-%m-%d} / {mode}: MAE={np.abs(actual - quantiles[:, 1]).mean():.3f}", flush=True)

    results = AREA / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / "chronos2.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    metadata["predictions_sha256"] = sha256(csv_path.read_bytes()).hexdigest()
    (results / "chronos2.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
