"""Validate complete participation, publish bundles, and prepare notebook tables."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from .tasks import (
    AREA,
    DATASETS,
    baselines,
    task_contract,
    forecast_window,
    load_data,
    require,
    select_cycles,
)
from .metrics import aggregate_scores, comparison, measure


@dataclass(frozen=True)
class Prediction:
    point: np.ndarray
    point_kind: str = "unspecified"
    quantiles: dict | None = None


def validate_prediction(prediction, horizon, levels):
    require(
        prediction.point_kind in ("mean", "median", "unspecified"), "invalid point interpretation"
    )
    point = np.asarray(prediction.point)
    require(point.shape == (horizon,) and np.isfinite(point).all(), "invalid point forecast")
    quantiles = prediction.quantiles or {}
    require(set(quantiles) == set(levels), "quantiles differ from declared outputs")
    if quantiles:
        require(set(quantiles) == {0.1, 0.5, 0.9}, "expected p10/p50/p90")
        values = np.stack([quantiles[q] for q in sorted(quantiles)], axis=-1)
        require(
            values.shape == (horizon, 3) and np.isfinite(values).all(),
            "invalid quantile shape or values",
        )
        require((np.diff(values, axis=1) >= 0).all(), "unordered quantiles")
        if prediction.point_kind == "median":
            require(np.allclose(point, quantiles[0.5], rtol=0, atol=0), "point differs from median")


def bundle_path(task, model, root=None):
    return (root or AREA / "results") / task.id / task.default_split / model


def load_bundle(task, path, *, area=AREA, allow_smoke=False, data=None, cycles=None):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    require(meta["schema_version"] == 5, "unsupported result format: expected schema v5")
    require(
        meta["task_id"] == task.id and meta["split"] == task.default_split,
        "incorrect task or split",
    )
    require(not meta["smoke"] or allow_smoke, "smoke results cannot enter comparisons")
    require(meta["task_contract"] == task_contract(task), "stale task or input contract")
    require(
        sha256((area / "datasets" / DATASETS[task.dataset].file).read_bytes()).hexdigest()
        == meta["source_sha256"],
        "stale dataset",
    )
    csv = path / "predictions.csv"
    require(
        sha256(csv.read_bytes()).hexdigest() == meta["predictions_sha256"],
        "prediction file changed",
    )
    if cycles is None:
        cycles = select_cycles(task, smoke=meta["smoke"], area=area)
    recorded = pd.DataFrame(meta["schedule"])
    recorded["forecast_start"] = pd.to_datetime(
        recorded.forecast_start, utc=DATASETS[task.dataset].utc
    )
    pd.testing.assert_frame_equal(recorded, cycles, check_dtype=False)
    model_id = meta["model"]["spec"]["id"]
    require(bool(model_id) and not model_id.startswith("baseline:"), "invalid model ID")
    result = pd.read_csv(csv, dtype={"entity": str})
    keys = ["entity", "track", "forecast_start"]
    required = {*keys, "timestamp", "step", "actual", "point", "point_kind"}
    require(required.issubset(result.columns), "missing prediction columns")
    for col in ["forecast_start", "timestamp"]:
        result[col] = pd.to_datetime(result[col], utc=DATASETS[task.dataset].utc)
    require(not result.duplicated([*keys, "step"]).any(), "duplicate predictions")
    quantiles = meta["model"]["spec"]["quantiles"]
    require(quantiles in ([], [0.1, 0.5, 0.9]), "incorrect quantile levels")
    qcols = [f"p{int(q * 100)}" for q in quantiles]
    require(
        set(c for c in result if c in ["p10", "p50", "p90"]) == set(qcols),
        "incorrect quantile columns",
    )
    expected = {
        (str(c.entity), track, c.forecast_start)
        for c in cycles.itertuples(index=False) for track in task.tracks
    }
    timings = pd.DataFrame(meta["timings"], columns=[*keys, "seconds"])
    timings["forecast_start"] = pd.to_datetime(timings.forecast_start, utc=DATASETS[task.dataset].utc)
    require(not timings.duplicated(keys).any() and set(timings[keys].itertuples(index=False, name=None)) == expected,
            "incomplete timing records")
    require(np.isfinite(timings.seconds.to_numpy(dtype=float)).all() and timings.seconds.ge(0).all(), "invalid timing")
    groups = dict(tuple(result.groupby(keys, sort=False)))
    require(set(groups) == expected, "incomplete prediction scenarios")
    if data is None:
        data = load_data(task, area)
    for cycle in cycles.itertuples(index=False):
        _, future = forecast_window(data, cycle, task)
        expected_rows = future[["timestamp", "target"]].rename(columns={"target": "actual"})
        expected_rows = expected_rows.reset_index(drop=True).assign(step=np.arange(1, task.horizon + 1))
        for track in task.tracks:
            group = groups[(cycle.entity, track, cycle.forecast_start)]
            require(group.point_kind.nunique() == 1, "inconsistent point interpretation")
            validate_prediction(Prediction(group.point.to_numpy(), group.point_kind.iloc[0],
                                {q: group[f"p{int(100 * q)}"].to_numpy() for q in quantiles}),
                                task.horizon, quantiles)
            pd.testing.assert_frame_equal(
                group[["timestamp", "actual", "step"]].reset_index(drop=True), expected_rows,
                check_dtype=False, check_exact=False, rtol=0, atol=1e-9,
            )
    return result, meta


def write_bundle(task, path, frame, metadata, *, area=AREA):
    """Stage and validate before replacing; restore the old bundle on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".staging-", dir=path.parent))
    backup = stage.with_name(stage.name.replace(".staging-", ".backup-"))
    try:
        frame.to_csv(stage / "predictions.csv", index=False)
        metadata = dict(
            metadata,
            predictions_sha256=sha256((stage / "predictions.csv").read_bytes()).hexdigest(),
        )
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        load_bundle(task, stage, area=area, allow_smoke=True)
        if path.exists():
            os.replace(path, backup)
        try:
            os.replace(stage, path)
        except BaseException:
            if backup.exists():
                os.replace(backup, path)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@dataclass
class Comparison:
    scores: pd.DataFrame
    summary: pd.DataFrame
    entities: pd.DataFrame
    predictions: pd.DataFrame
    timings: pd.DataFrame
    provenance: pd.DataFrame
    issues: list
    windows: dict
    data: pd.DataFrame
    schedule: pd.DataFrame


def load_comparison(task, area=AREA, *, strict=False):
    data, cycles = load_data(task, area), select_cycles(task, area=area)
    windows, rows, frames, timings, provenance, issues = {}, [], [], [], [], []

    def score(model_id, name, track, entity, start, point, quantiles=None):
        history, future, _ = windows[(entity, start)]
        rows.append({
            "Model ID": model_id,
            "Model": name,
            "Track": track,
            "entity": entity,
            "forecast_start": start,
            **measure(future.target, point, history.target, task, quantiles),
        })

    for cycle in cycles.itertuples(index=False):
        history, future = forecast_window(data, cycle, task)
        alternatives = baselines(history, task)
        windows[(cycle.entity, cycle.forecast_start)] = (history, future, alternatives)
        for track in task.tracks:
            for name, values in alternatives.items():
                score(f"baseline:{name}", name, track, cycle.entity, cycle.forecast_start, values)
    directory = area / "results" / task.id / task.default_split
    for path in sorted(directory.glob("*/metadata.json")):
        if path.parent.name.startswith("."):
            continue
        try:
            frame, meta = load_bundle(task, path.parent, area=area, data=data, cycles=cycles)
            require(meta["model"]["spec"]["id"] == path.parent.name, "bundle directory differs from model ID")
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            if strict:
                raise
            issues.append(f"{path.parent.name}: {error}")
            continue
        model_id, name = meta["model"]["spec"]["id"], meta["model"]["spec"]["name"]
        for (entity, start, track), group in frame.groupby(["entity", "forecast_start", "track"], sort=False):
            quantiles = {q: group[f"p{int(100 * q)}"].to_numpy()
                         for q in meta["model"]["spec"]["quantiles"]} or None
            score(model_id, name, track, entity, start, group.point, quantiles)
        frames.append(frame.assign(**{"Model ID": model_id, "Model": name}))
        timings.extend({"Model ID": model_id, "Model": name, "Track": w["track"], "Seconds": w["seconds"]}
                       for w in meta["timings"])
        provenance.append({
            "Model ID": model_id,
            "Model": name,
            "Checkpoint": meta["model"]["spec"]["revision"],
            "Library": meta["model"].get("library_version", "unknown"),
            "Generated (UTC)": meta["generated_at"],
        })
    scores = pd.DataFrame(rows)
    summary, entities = aggregate_scores(scores)
    return Comparison(scores, comparison(summary, task), entities,
                      pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
                      pd.DataFrame(timings), pd.DataFrame(provenance), issues, windows, data, cycles)


def main():
    """Rebuild tables from saved predictions without model inference."""
    import argparse
    from .tasks import TASKS, TRACKS, task_label

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    sections = ["# Saved forecasting comparisons", "",
                "Validated schema-v5 results. Each track uses its full schedule. Lower scores are better; "
                "improvement compares aggregate primary scores with the fixed reference. "
                "Zero-shot does not establish absence from pretraining. CPU timings exclude loading/downloads.", ""]
    for task in TASKS.values():
        view = load_comparison(task, strict=True)
        sections.extend([f"## {task_label(task)}", "",
                         f"Split **{task.default_split}** · reference **{task.reference}** · main score **{task.primary_metric}**.", ""])
        if view.issues:
            sections.extend(["Unavailable participation: " + "; ".join(view.issues), ""])
        columns = ["Model ID", "Model", "MAE", "MASE", "Improvement (%)", "WIS", "Coverage80 (%)", "Windows"]
        columns = [column for column in columns if column in view.summary]
        for track in task.tracks:
            selected = view.summary[view.summary.Track == track].sort_values([task.primary_metric, "Model ID"])
            sections.extend([f"### {TRACKS[track]}", "", "| " + " | ".join(columns) + " |",
                             "| " + " | ".join("---" for _ in columns) + " |"])
            for row in selected[columns].itertuples(index=False, name=None):
                values = ["—" if pd.isna(v) else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                sections.append("| " + " | ".join(values) + " |")
            sections.append("")
    markdown = "\n".join(sections) + "\n"
    if args.markdown:
        args.markdown.write_text(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
