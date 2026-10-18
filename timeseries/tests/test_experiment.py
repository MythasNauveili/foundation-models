"""Behavioral checks use the lightweight environment; no model weights or imports."""

from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from foundation_timeseries import tasks as experiment
from foundation_timeseries.tasks import (
    AREA,
    TASKS,
    DATASETS,
    task_contract,
    task_label,
    build_input,
    TEACHING_TASKS,
    forecast_window,
    load_data,
    select_cycles,
)
from foundation_timeseries.runner import (
    ModelSpec,
    Prediction,
    run,
    validate_prediction,
    argument_parser,
    selected_tasks,
)
from foundation_timeseries.results import load_bundle, write_bundle, load_comparison
from foundation_timeseries.metrics import (
    mae,
    bias,
    measure,
    seasonal_scale,
    interval_metrics,
    aggregate_scores,
)

RTE = TASKS["rte-hourly-24-v1"]
WEATHER = TASKS["weather-daily-7-v1"]
_spec = importlib.util.spec_from_file_location("prepare_rte", AREA / "datasets" / "prepare_rte.py")
prepare_rte = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_rte)


class BaselineTests(unittest.TestCase):
    def test_missing_references_and_seasonal_repetition(self):
        np.testing.assert_array_equal(
            experiment.persistence([2, np.nan, 4, np.nan], 3), [4, 4, 4]
        )
        self.assertTrue(np.isnan(experiment.persistence([np.nan], 3)).all())
        np.testing.assert_array_equal(
            experiment.seasonal_naive([1, 2, 3, np.nan], 5, lag=2), [3, 2, 3, 2, 3]
        )
        self.assertTrue(np.isnan(experiment.seasonal_naive([np.nan, 2], 2, lag=2)[0]))
        self.assertTrue(np.isnan(experiment.seasonal_naive([2], 2, lag=4)).all())

    def test_four_week_average_uses_only_four_references(self):
        np.testing.assert_allclose(
            experiment.seasonal_average([999, 1, 20, 2, 30, 3, np.nan, 4, 50, 5], 2, lag=2, seasons=4),
            [100 / 3, 3.5],
        )

    def test_window_entity_isolation_and_future_independence(self):
        timestamps = pd.date_range("2024-01-01", periods=8, freq="D")
        data = pd.concat(
            [
                pd.DataFrame(
                    {"entity": entity, "timestamp": timestamps, "target": np.arange(8) + offset}
                )
                for entity, offset in [("a", 0), ("b", 100)]
            ],
            ignore_index=True,
        )
        task = replace(WEATHER, history_length=4, horizon=3)
        cycle = SimpleNamespace(entity="a", forecast_start=timestamps[5])
        history, future = forecast_window(data, cycle, task)
        before = build_input(history, future.timestamp, task, "target_only")
        data.loc[data.timestamp >= timestamps[5], "target"] = 9000
        changed, future = forecast_window(data, cycle, task)
        after = build_input(changed, future.timestamp, task, "target_only")
        np.testing.assert_array_equal(before.target, [1, 2, 3, 4])
        np.testing.assert_array_equal(before.target, after.target)
        self.assertFalse(hasattr(before, "actual"))
        self.assertFalse(hasattr(before, "future_target"))
        with self.assertRaisesRegex(ValueError, "insufficient history"):
            forecast_window(data, cycle, replace(task, history_length=6))


class MetricTests(unittest.TestCase):
    def test_point_metrics(self):
        self.assertEqual(mae([10, 10], [8, 12]), 2)
        self.assertEqual(bias([10, 10], [8, 12]), 0)

    def test_scales_use_only_valid_historical_pairs(self):
        self.assertEqual(measure([100], [268], np.arange(336), RTE)["MASE"], 1)
        self.assertTrue(np.isnan(seasonal_scale([1, 1, 1], 1)))
        self.assertTrue(np.isnan(seasonal_scale([1], 2)))
        self.assertEqual(seasonal_scale([1, np.nan, 4, 7], 1), 3)

    def test_interval_score_penalizes_misses_and_width(self):
        values = interval_metrics([0, 5], [-1, 1], [0, 2], [1, 3])
        self.assertAlmostEqual(values["WIS"], (0.2 / 1.5 + 3.7 / 1.5) / 2)
        self.assertEqual(values["Coverage80 (%)"], 50)
        self.assertEqual(values["Width80"], 2)

    def test_equal_entity_aggregation_and_undefined_scores(self):
        scores = pd.DataFrame(
            {"entity": ["a", "a", "b"], "Model ID": "test", "Model": "test", "Track": "target_only", "MAE": [1, 3, 10]}
        )
        self.assertEqual(aggregate_scores(scores)[0].MAE.iloc[0], 6)
        scores.loc[0, "MAE"] = np.nan
        self.assertTrue(np.isnan(aggregate_scores(scores)[0].MAE.iloc[0]))


class InputTests(unittest.TestCase):
    def test_all_track_memberships_and_lengths(self):
        for task in TASKS.values():
            data = load_data(task)
            cycle = next(select_cycles(task).itertuples(index=False))
            history, future = forecast_window(data, cycle, task)
            spec = DATASETS[task.dataset]
            for track in task.tracks:
                inputs = build_input(history, future.timestamp, task, track)
                past = set(spec.past_columns) if track.startswith("past_observed") else set()
                known = (
                    set(spec.calendar_columns)
                    if track in ("calendar", "past_observed_calendar")
                    else set()
                )
                self.assertEqual(set(inputs.past_covariates), past)
                self.assertEqual(set(inputs.known_history), known)
                self.assertEqual(set(inputs.known_future), known)
                self.assertTrue(
                    all(
                        len(v) == task.history_length
                        for v in [
                            inputs.target,
                            *inputs.past_covariates.values(),
                            *inputs.known_history.values(),
                        ]
                    )
                )
                self.assertTrue(all(len(v) == task.horizon for v in inputs.known_future.values()))

    def test_rte_calendar_uses_local_clock_over_dst(self):
        dates = pd.Series(pd.date_range("2024-03-31T00:00:00Z", periods=3, freq="h"))
        values = experiment.calendar_features(dates, ("hour", "weekday"), "Europe/Paris")
        self.assertEqual(values["hour"].tolist(), [1, 3, 4])
        self.assertEqual(values["weekday"].tolist(), [6, 6, 6])

    def test_preparation_eligibility_does_not_require_future_weather(self):
        spec = importlib.util.spec_from_file_location("prepare", AREA / "datasets" / "prepare.py")
        preparation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparation)
        tasks = [t for t in TASKS.values() if t.default_split == "exploratory"]
        frames = {t.dataset: load_data(t).set_index("timestamp") for t in tasks}
        for task in tasks:
            columns = DATASETS[task.dataset].past_columns
            if columns:
                first = select_cycles(task).iloc[0].forecast_start
                frames[task.dataset].loc[first, columns[0]] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(preparation, "DESTINATION", Path(directory)):
                preparation.save_cycles(frames)
                actual = pd.read_csv(Path(directory) / "cycles.csv", parse_dates=["forecast_start"])
                for task in tasks:
                    expected = select_cycles(task)
                    self.assertEqual(
                        actual.loc[actual.task == task.id, "forecast_start"].tolist(),
                        expected.forecast_start.tolist(),
                    )
                first = select_cycles(WEATHER).iloc[0].forecast_start
                frames["weather"].loc[first, "target"] = np.nan
                with self.assertRaises(AssertionError):
                    preparation.save_cycles(frames)

    def test_reserved_split_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            run(
                [replace(RTE, default_split="evaluation")],
                DUMMY_SPEC,
                lambda: self.fail("loaded reserved model"),
            )

    def test_cli_defaults_and_selection(self):
        parser = argument_parser("test", ["5M", "28M"])
        args = parser.parse_args([])
        self.assertEqual(args.size, "5M")
        self.assertEqual(len(selected_tasks(args)), len(TASKS))
        self.assertEqual(
            len(selected_tasks(parser.parse_args(["--suite", "teaching"]))),
            len(TEACHING_TASKS),
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--task", RTE.id, "--suite", "showcase"])

    def test_contract_changes_only_for_semantic_inputs(self):
        before = task_contract(WEATHER)
        with patch.dict(TASKS, {"unrelated": replace(RTE, id="unrelated")}):
            self.assertEqual(task_contract(WEATHER), before)
        self.assertNotEqual(task_contract(replace(WEATHER, horizon=8)), before)
        with patch.dict(
            DATASETS, {"weather": replace(DATASETS["weather"], calendar_columns=("month",))}
        ):
            self.assertNotEqual(task_contract(WEATHER), before)

    def test_cosmetic_definitions_preserve_contract_and_labels_distinguish_tasks(self):
        before = task_contract(WEATHER)
        changed = replace(DATASETS["weather"], title="A clearer title", card="new-card.md", unit="degrees")
        with patch.dict(DATASETS, {"weather": changed}):
            self.assertEqual(task_contract(WEATHER), before)
        with patch.object(experiment, "INPUT_VERSION", experiment.INPUT_VERSION + 1):
            self.assertNotEqual(task_contract(WEATHER), before)
        other = replace(WEATHER, id="weather-daily-3-v1", horizon=3)
        self.assertNotEqual(task_label(WEATHER), task_label(other))
        self.assertIn("3 steps", task_label(other))

    def test_preparation_supports_two_tasks_per_dataset_and_preserves_unowned_rows(self):
        spec = importlib.util.spec_from_file_location("prepare", AREA / "datasets/prepare.py")
        preparation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparation)
        other = replace(WEATHER, id="weather-daily-3-v1", horizon=3)
        frames = {t.dataset: load_data(t).set_index("timestamp") for t in TASKS.values() if t.id in TEACHING_TASKS}
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation, "DESTINATION", Path(directory)):
            pd.DataFrame([dict(task="unowned-weather-task", entity="single", forecast_start="2025-01-01", split="development")]).to_csv(Path(directory) / "cycles.csv", index=False)
            starts = {key: select_cycles(TASKS[key]).forecast_start.tolist() for key in TEACHING_TASKS}
            starts[other.id] = starts[WEATHER.id]
            with patch.dict(TASKS, {other.id: other}):
                preparation.save_cycles(frames, starts)
            actual = pd.read_csv(Path(directory) / "cycles.csv")
            self.assertEqual(actual.columns.tolist(), experiment.SCHEDULE_COLUMNS)
            self.assertEqual(sum(actual.task == WEATHER.id), 12)
            self.assertEqual(sum(actual.task == other.id), 12)
            self.assertEqual(sum(actual.task == "unowned-weather-task"), 1)
            for task in (WEATHER, other):
                self.assertEqual(len(experiment.normalize_cycles(actual, task)), 12)


DUMMY_SPEC = ModelSpec("dummy", "Dummy", "none", "fixed", "none", ".", quantiles=())


class DummyAdapter:
    spec = DUMMY_SPEC

    def predict(self, inputs):
        assert not hasattr(inputs, "actual")
        assert not hasattr(inputs, "future_target")
        assert not set(inputs.known_future).intersection(DATASETS[WEATHER.dataset].past_columns)
        return Prediction(np.full(inputs.horizon, inputs.target[-1]))


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.area = Path(self.directory.name)
        (self.area / "datasets").mkdir()
        for name in ["cycles.csv", "strasbourg_daily.csv"]:
            shutil.copy(AREA / "datasets" / name, self.area / "datasets" / name)
        self.provenance = {"spec": asdict(DUMMY_SPEC)}

    def generate(self, adapter=DummyAdapter, spec=DUMMY_SPEC):
        return run([WEATHER], spec, adapter, area=self.area, provenance=self.provenance)[0]

    def test_point_only_model_is_ranked_and_matching_run_skips_loading(self):
        path = self.generate()
        frame, meta = load_bundle(WEATHER, path, area=self.area)
        self.assertEqual(len(frame), 12 * 7 * 4)
        self.assertNotIn("p50", frame)
        run(
            [WEATHER],
            DUMMY_SPEC,
            lambda: self.fail("matching run loaded weights"),
            area=self.area,
            provenance=self.provenance,
        )
        view = load_comparison(WEATHER, self.area)
        self.assertIn("Dummy", set(view.summary.Model))
        self.assertEqual(view.issues, [])

    def test_comparison_loads_dataset_and_schedule_once(self):
        self.generate()
        with patch("foundation_timeseries.results.load_data", wraps=load_data) as data_loader, \
             patch("foundation_timeseries.results.select_cycles", wraps=select_cycles) as schedule_loader:
            load_comparison(WEATHER, self.area)
        self.assertEqual(data_loader.call_count, 1)
        self.assertEqual(schedule_loader.call_count, 1)

    def test_failed_track_aborts_without_publishing_partial_results(self):
        calls = []

        class Failing(DummyAdapter):
            def predict(self, inputs):
                calls.append(inputs)
                if inputs.known_future:
                    raise RuntimeError("calendar failed")
                return super().predict(inputs)

        with self.assertRaisesRegex(RuntimeError, "dummy / weather-daily-7-v1 / calendar / single / 2025"):
            self.generate(Failing)
        self.assertEqual(len(calls), len(select_cycles(WEATHER)) + 1)
        self.assertFalse((self.area / "results").exists())

    def test_stale_source_task_and_incomplete_exports_are_rejected(self):
        path = self.generate()
        frame, meta = load_bundle(WEATHER, path, area=self.area)
        original = (path / "predictions.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid point forecast|step"):
            write_bundle(WEATHER, path, frame.iloc[1:], meta, area=self.area)
        self.assertEqual((path / "predictions.csv").read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "stale task"):
            load_bundle(replace(WEATHER, horizon=8), path, area=self.area)
        source = self.area / "datasets" / "strasbourg_daily.csv"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "stale dataset"):
            load_bundle(WEATHER, path, area=self.area)

    def test_interrupted_replace_restores_previous_bundle(self):
        path = self.generate()
        frame, meta = load_bundle(WEATHER, path, area=self.area)
        import os

        original_replace = os.replace

        def interrupt(source, destination):
            if Path(source).name.startswith(".staging-") and Path(destination) == path:
                raise KeyboardInterrupt("interrupted publish")
            return original_replace(source, destination)

        with patch("foundation_timeseries.results.os.replace", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                write_bundle(WEATHER, path, frame, meta, area=self.area)
        load_bundle(WEATHER, path, area=self.area)
        self.assertFalse(list(path.parent.glob(".staging-*")))
        self.assertFalse(list(path.parent.glob(".backup-*")))

    def test_incompatible_context_fails_before_inference_and_preserves_results(self):
        path = self.generate()
        original = (path / "predictions.csv").read_bytes()

        class TooShort(DummyAdapter):
            runtime = {"max_context": 10}

            def predict(self, inputs):
                raise AssertionError("incompatible context reached inference")

        with self.assertRaisesRegex(ValueError, "dummy / weather.*context exceeds model limit"):
            run([WEATHER], DUMMY_SPEC, TooShort, area=self.area, refresh=True)
        self.assertEqual((path / "predictions.csv").read_bytes(), original)
        load_bundle(WEATHER, path, area=self.area)

    def test_smoke_bundle_never_enters_full_comparison(self):
        path = run(
            [WEATHER],
            DUMMY_SPEC,
            DummyAdapter,
            area=self.area,
            smoke=True,
            provenance=self.provenance,
        )[0]
        self.addCleanup(shutil.rmtree, path.parents[2])
        with self.assertRaisesRegex(ValueError, "smoke results"):
            load_bundle(WEATHER, path, area=self.area)
        frame, _ = load_bundle(WEATHER, path, area=self.area, allow_smoke=True)
        self.assertEqual(len(frame), WEATHER.horizon * len(WEATHER.tracks))

    def test_failed_refresh_keeps_previous_valid_bundle(self):
        path = self.generate()
        original = (path / "predictions.csv").read_bytes()

        class Broken(DummyAdapter):
            def predict(self, inputs):
                raise RuntimeError("failed refresh")

        with self.assertRaisesRegex(RuntimeError, "failed refresh"):
            run([WEATHER], DUMMY_SPEC, Broken, area=self.area, refresh=True, provenance=self.provenance)
        self.assertEqual((path / "predictions.csv").read_bytes(), original)
        load_bundle(WEATHER, path, area=self.area)

    def test_invalid_quantiles_cannot_be_exported(self):
        spec = replace(DUMMY_SPEC, quantiles=(0.1, 0.5, 0.9))
        with self.assertRaisesRegex(ValueError, "unordered quantiles"):
            validate_prediction(
                Prediction(
                    np.ones(2),
                    "median",
                    {0.1: np.ones(2) * 2, 0.5: np.ones(2), 0.9: np.ones(2) * 3},
                ),
                2,
                spec.quantiles,
            )
        with self.assertRaisesRegex(ValueError, "invalid point"):
            validate_prediction(Prediction(np.array([np.nan])), 1, DUMMY_SPEC.quantiles)

    def test_same_display_name_does_not_merge_models(self):
        self.generate()
        other = replace(DUMMY_SPEC, id="other-dummy")

        class Other(DummyAdapter):
            spec = other

        run([WEATHER], other, Other, area=self.area)
        view = load_comparison(WEATHER, self.area)
        models = view.summary[view.summary.Model == "Dummy"]
        self.assertEqual(set(models["Model ID"]), {"dummy", "other-dummy"})
        self.assertEqual(len(models), 2 * len(WEATHER.tracks))
        self.assertTrue(models.Windows.eq(12).all())

    def test_cosmetic_changes_resume_but_schedule_and_provenance_changes_do_not(self):
        path = self.generate()
        with patch.dict(DATASETS, {"weather": replace(DATASETS["weather"], title="Teaching title")}):
            load_bundle(WEATHER, path, area=self.area)
            run([WEATHER], DUMMY_SPEC, lambda: self.fail("cosmetic change loaded model"), area=self.area, provenance=self.provenance)
        changed_provenance = dict(self.provenance, library_version="changed")
        with self.assertRaisesRegex(RuntimeError, "provenance mismatch loaded model"):
            def load():
                raise RuntimeError("provenance mismatch loaded model")
            run([WEATHER], DUMMY_SPEC, load, area=self.area, provenance=changed_provenance)
        schedule = self.area / "datasets/cycles.csv"
        cycles = pd.read_csv(schedule)
        cycles = cycles.drop(cycles.index[cycles.task == WEATHER.id][0])
        cycles.to_csv(schedule, index=False)
        with self.assertRaises(AssertionError):
            load_bundle(WEATHER, path, area=self.area)

    def test_cli_failure_exits_unsuccessfully_with_window_context(self):
        code = f"""
from foundation_timeseries.runner import ModelSpec, run
from foundation_timeseries.tasks import TASKS
from pathlib import Path
class Broken:
    spec = ModelSpec('dummy', 'Dummy', 'none', 'fixed', 'none', '.', quantiles=())
    def predict(self, inputs):
        raise ValueError('injected CLI failure')
run([TASKS['weather-daily-7-v1']], Broken.spec, Broken, area=Path({str(self.area)!r}))
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dummy / weather-daily-7-v1 / target_only / single / 2025", result.stderr)
        self.assertIn("injected CLI failure", result.stderr)
        self.assertFalse((self.area / "results").exists())

    def test_irregular_source_grid_and_invalid_saved_bundle_are_rejected(self):
        path = self.generate()
        frame = pd.read_csv(self.area / "datasets/strasbourg_daily.csv")
        frame.drop(index=1).to_csv(self.area / "datasets/strasbourg_daily.csv", index=False)
        with self.assertRaisesRegex(ValueError, "irregular timestamps"):
            load_data(WEATHER, self.area)
        shutil.copy(AREA / "datasets/strasbourg_daily.csv", self.area / "datasets/strasbourg_daily.csv")
        metadata_path = path / "metadata.json"
        meta = json.loads(metadata_path.read_text())
        meta["schema_version"] = 3
        metadata_path.write_text(json.dumps(meta))
        self.assertTrue(load_comparison(WEATHER, self.area).issues)
        with self.assertRaisesRegex(ValueError, "unsupported result format"):
            load_comparison(WEATHER, self.area, strict=True)

    def test_invalid_prediction_aborts_before_publication(self):
        calls = []

        class Invalid(DummyAdapter):
            def predict(self, inputs):
                calls.append(inputs)
                return Prediction(np.full(inputs.horizon, np.nan))

        with self.assertRaisesRegex(RuntimeError, "dummy / weather-daily-7-v1 / target_only / single / .*invalid point"):
            self.generate(Invalid)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.area / "results").exists())

    def test_saved_predictions_keep_alignment_and_completeness_checks(self):
        path = self.generate()
        frame, meta = load_bundle(WEATHER, path, area=self.area)
        for column, value, message in [("timestamp", pd.Timestamp("1900-01-01"), "timestamp"),
                                       ("actual", -1000, "actual"), ("step", 999, "step")]:
            changed = frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(AssertionError, message):
                write_bundle(WEATHER, path, changed, meta, area=self.area)
        changed = dict(meta, timings=meta["timings"][1:])
        with self.assertRaisesRegex(ValueError, "incomplete timing"):
            write_bundle(WEATHER, path, frame, changed, area=self.area)
        with self.assertRaisesRegex(ValueError, "incomplete prediction"):
            write_bundle(WEATHER, path, frame[frame.track != "calendar"], meta, area=self.area)
        load_bundle(WEATHER, path, area=self.area)


class RteTests(unittest.TestCase):
    def test_utc_dst_continuity_and_missing_half_hour(self):
        for start in ["2024-03-31T00:00:00Z", "2024-10-27T00:00:00Z"]:
            timestamps = pd.date_range(start, periods=8, freq="30min")
            raw = pd.DataFrame(
                {
                    "code_insee_region": "44",
                    "date_heure": timestamps,
                    "consommation": [10, 30, 20, np.nan, 40, 60, 80, 100],
                }
            )
            end = timestamps[-1] + pd.Timedelta(minutes=30)
            hourly = prepare_rte.hourly_consumption(raw, timestamps[0], end)
            self.assertEqual(len(hourly), 4)
            np.testing.assert_equal(hourly.target.to_numpy(), [20, np.nan, 50, 90])
            self.assertTrue(hourly.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all())
            absent = prepare_rte.hourly_consumption(raw.drop(index=1), timestamps[0], end)
            self.assertTrue(pd.isna(absent.target.iloc[0]))

    def test_only_nonexistent_spring_clock_labels_are_discarded(self):
        raw = pd.DataFrame(
            {
                "code_insee_region": ["44", "44"],
                "date_heure": ["2024-03-31T01:00:00Z"] * 2,
                "date": ["2024-03-31"] * 2,
                "heure": ["02:00", "03:00"],
                "consommation": [99, 10],
            }
        )
        corrected, notes = prepare_rte.correct_spring_labels(raw)
        self.assertEqual(corrected.consommation.tolist(), [10])
        self.assertEqual(notes["discarded_nonexistent_local_clock_rows"], 1)
        # An ordinary Sunday mismatch is not a permitted DST correction.
        raw["date_heure"], raw["date"] = "2024-03-10T02:00:00Z", "2024-03-10"
        with self.assertRaisesRegex(AssertionError, "unexpected source clock"):
            prepare_rte.correct_spring_labels(raw)

    def test_frozen_extract_and_reserved_schedule(self):
        data, cycles = experiment.load_data(RTE), experiment.load_cycles(RTE)
        meta = json.loads((experiment.AREA / "datasets" / "rte_hourly.json").read_text())
        self.assertEqual(
            sha256((experiment.AREA / "datasets" / "rte_hourly.csv").read_bytes()).hexdigest(),
            meta["dataset_sha256"],
        )
        self.assertEqual(set(data.entity), set(experiment.REGIONS))
        self.assertEqual(int(data.target.isna().sum()), 12)
        self.assertEqual(
            cycles.groupby("split").size().to_dict(), {"development": 120, "evaluation": 126}
        )
        self.assertFalse(meta["reserved_scores_computed"])
        for cycle in cycles[cycles.split == "development"].itertuples(index=False):
            history, future = experiment.forecast_window(data, cycle, RTE)
            self.assertEqual(len(history), 1344)
            self.assertEqual(len(future), 24)
            self.assertEqual(future.timestamp.dt.year.unique().tolist(), [2024])

    def test_conflicting_duplicates_stop_preparation(self):
        raw = pd.DataFrame(
            {
                "code_insee_region": ["44", "44"],
                "date_heure": ["2024-01-01T00:00:00Z"] * 2,
                "consommation": [10, 20],
            }
        )
        with self.assertRaisesRegex(AssertionError, "conflicting"):
            prepare_rte.hourly_consumption(raw, "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    def test_fixed_schedule_exclusions_without_replacement(self):
        times = pd.date_range(prepare_rte.START, prepare_rte.END, freq="h", inclusive="left")
        data = pd.DataFrame({"entity": "44", "timestamp": times, "target": 10.0})
        data.loc[data.timestamp.eq(pd.Timestamp("2024-01-08T04:00:00Z")), "target"] = np.nan
        eligible, excluded = prepare_rte.rte_cycles(data)
        self.assertEqual(len(eligible) + len(excluded), 6 * 48)
        self.assertFalse(
            (
                (eligible.entity == "44") & eligible.forecast_start.eq("2024-01-08T00:00:00+00:00")
            ).any()
        )
        self.assertTrue(
            any(
                row["entity"] == "44" and row["forecast_start"].startswith("2024-01-08")
                for row in excluded
            )
        )
        self.assertTrue(eligible.entity.eq("44").all())
        self.assertEqual(set(pd.to_datetime(eligible.forecast_start).dt.day), {8, 22})


if __name__ == "__main__":
    unittest.main()
