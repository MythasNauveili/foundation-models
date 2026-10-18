"""Run each model on the same declared histories, inputs, and forecast windows."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import platform
from tempfile import mkdtemp
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .tasks import (
    AREA, DATASETS, TASKS, TEACHING_TASKS, build_input, forecast_window,
    load_data, require, select_cycles, task_contract,
)
from .results import Prediction, bundle_path, load_bundle, validate_prediction, write_bundle


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    checkpoint: str
    revision: str
    library: str
    environment: str
    quantiles: tuple = (0.1, 0.5, 0.9)
    settings: tuple = ()


def model_provenance(spec, area=AREA):
    return {
        "spec": asdict(spec),
        "library_version": version(spec.library),
        "lockfile_sha256": sha256((area / spec.environment / "uv.lock").read_bytes()).hexdigest(),
    }


def forecast_task(adapter, task, data, schedule):
    """History → permitted inputs → native prediction → validated forecast rows."""
    limit = getattr(adapter, "runtime", {}).get("max_context")
    require(limit is None or task.history_length <= limit,
            f"{adapter.spec.id} / {task.id}: context exceeds model limit ({limit})")
    rows, timings = [], []
    for track in task.tracks:
        for cycle in schedule.itertuples(index=False):
            try:
                history, future = forecast_window(data, cycle, task)
                inputs = build_input(history, future.timestamp, task, track)
                started = perf_counter()
                prediction = adapter.predict(inputs)
                elapsed = perf_counter() - started
                validate_prediction(prediction, task.horizon, adapter.spec.quantiles)
            except Exception as error:
                raise RuntimeError(
                    f"{adapter.spec.id} / {task.id} / {track} / {cycle.entity} / "
                    f"{cycle.forecast_start}: {error}"
                ) from error
            keys = {"entity": cycle.entity, "track": track, "forecast_start": str(cycle.forecast_start)}
            timings.append({**keys, "seconds": elapsed})
            rows.append(pd.DataFrame({
                **keys, "timestamp": future.timestamp.to_numpy(),
                "step": np.arange(1, task.horizon + 1), "actual": future.target.to_numpy(),
                "point": np.asarray(prediction.point, dtype=float), "point_kind": prediction.point_kind,
                **{f"p{int(100 * q)}": values for q, values in (prediction.quantiles or {}).items()},
            }))
        print(f"{adapter.spec.name} / {task.id} / {track}: {len(schedule)} windows", flush=True)
    return pd.concat(rows, ignore_index=True), timings


def describe_run(adapter, task, schedule, timings, provenance, *, area=AREA, smoke=False):
    """Record reproducible inputs and observed runtime details, without duplicating the model."""
    return {
        "schema_version": 5, "task_id": task.id, "split": task.default_split, "smoke": smoke,
        "task_contract": task_contract(task),
        "source_sha256": sha256((area / "datasets" / DATASETS[task.dataset].file).read_bytes()).hexdigest(),
        "schedule": schedule.assign(forecast_start=schedule.forecast_start.map(str)).to_dict("records"),
        "model": provenance, "runtime": getattr(adapter, "runtime", {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {"system": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), **getattr(adapter, "hardware", {})},
        "timings": timings,
    }


def run(tasks, spec, load_adapter, *, smoke=False, refresh=False, area=AREA, provenance=None):
    """Reuse matching results, or publish a complete task after every track succeeds."""
    require(not spec.id.startswith("baseline:"), "model ID uses reserved baseline prefix")
    schedules = {task.id: select_cycles(task, smoke=smoke, area=area) for task in tasks}
    provenance = json.loads(json.dumps(provenance or {"spec": asdict(spec)}))
    root = Path(mkdtemp(prefix="foundation-timeseries-smoke-")) if smoke else area / "results"
    adapter, paths = None, []
    for task in tasks:
        path = bundle_path(task, spec.id, root=root)
        if path.exists() and not (smoke or refresh):
            try:
                _, previous = load_bundle(task, path, area=area)
                if previous["model"] == provenance:
                    print(f"Skip matching complete bundle: {path}", flush=True)
                    paths.append(path)
                    continue
            except (OSError, ValueError, KeyError, TypeError, AssertionError):
                pass  # An invalid or stale bundle needs a complete replacement.
        if adapter is None:
            adapter = load_adapter()
        require(adapter.spec == spec, "adapter model differs from requested model")
        schedule = schedules[task.id]
        predictions, timings = forecast_task(adapter, task, load_data(task, area), schedule)
        metadata = describe_run(adapter, task, schedule, timings, provenance, area=area, smoke=smoke)
        write_bundle(task, path, predictions, metadata, area=area)
        print(f"Saved {len(predictions)} rows: {path}", flush=True)
        paths.append(path)
    return paths


def argument_parser(description, sizes=None):
    parser = argparse.ArgumentParser(description=description)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", action="append", choices=list(TASKS))
    selection.add_argument("--suite", choices=["teaching", "showcase"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    if sizes:
        parser.add_argument("--size", choices=["all", *sizes], default="5M")
    return parser


def selected_tasks(args):
    if args.task:
        return [TASKS[key] for key in dict.fromkeys(args.task)]
    return [TASKS[key] for key in TEACHING_TASKS] if args.suite == "teaching" else list(TASKS.values())
