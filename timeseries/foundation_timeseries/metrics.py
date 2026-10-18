"""Scores have explicit mathematical inputs, never dataset-name switches."""

import numpy as np
import pandas as pd


def mae(actual, forecast):
    return float(np.abs(np.asarray(forecast) - np.asarray(actual)).mean())


def bias(actual, forecast):
    return float((np.asarray(forecast) - np.asarray(actual)).mean())


def seasonal_scale(history, seasonal_period):
    target = np.asarray(history, dtype=float)
    pairs = np.abs(target[seasonal_period:] - target[:-seasonal_period])
    pairs = pairs[np.isfinite(pairs)]
    scale = pairs.mean() if len(pairs) else np.nan
    return float(scale) if np.isfinite(scale) and scale > 0 else np.nan


def interval_metrics(actual, p10, p50, p90):
    actual, lower, median, upper = map(np.asarray, (actual, p10, p50, p90))
    width = upper - lower
    interval = width + 10 * np.maximum(lower - actual, 0) + 10 * np.maximum(actual - upper, 0)
    return {
        "WIS": float(((0.5 * np.abs(actual - median) + 0.1 * interval) / 1.5).mean()),
        "Coverage80 (%)": float(100 * ((lower <= actual) & (actual <= upper)).mean()),
        "Width80": float(width.mean()),
    }


def measure(actual, point, history, task, quantiles=None):
    scale = seasonal_scale(history, task.seasonal_period)
    values = {
        "MAE": mae(actual, point),
        "Bias": bias(actual, point),
        "MASE": mae(actual, point) / scale,
    }
    if quantiles is not None:
        values.update(interval_metrics(actual, quantiles[0.1], quantiles[0.5], quantiles[0.9]))
        values["Scaled WIS"] = values["WIS"] / scale
    return values


def aggregate_scores(scores):
    groups = ["Model ID", "Model", "Track"]
    identifiers = {*groups, "entity", "forecast_start"}
    columns = [c for c in scores if c not in identifiers]
    regional = (
        scores.groupby([*groups, "entity"], sort=False)[columns]
        .agg(lambda x: x.mean(skipna=False))
        .reset_index()
    )
    overall = (
        regional.groupby(groups, sort=False)[columns]
        .agg(lambda x: x.mean(skipna=False))
        .reset_index()
    )
    counts = scores.groupby(groups).size().rename("Windows").reset_index()
    return overall.merge(counts, on=groups), regional


def comparison(summary, task):
    """Compare aggregate scores against the fixed baseline on the same windows."""
    metric = task.primary_metric
    output = summary.copy()
    reference = output.loc[output["Model ID"] == f"baseline:{task.reference}", metric].iloc[0]
    output["Improvement (%)"] = (
        100 * (1 - output[metric] / reference)
        if np.isfinite(reference) and reference > 0
        else np.nan
    )
    return output
