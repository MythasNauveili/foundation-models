"""The included showcase is inspectable without loading any model libraries."""

import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from foundation_timeseries.tasks import AREA, TASKS, select_cycles
from foundation_timeseries.results import load_bundle, load_comparison


# This refresh promises these four shipped tasks. New tasks can start with baselines
# or a subset of models without changing the shared experiment contract.
INCLUDED_TASKS = (
    "weather-daily-7-v1",
    "bikes-2hourly-12-v1",
    "retail-monthly-6-v1",
    "rte-hourly-24-v1",
)


class IncludedShowcaseTests(unittest.TestCase):
    def test_all_five_checkpoints_cover_all_declared_tracks(self):
        expected = {"chronos2", "timestral-5m", "timestral-28m", "timestral-123m", "timestral-209m"}
        rows = calls = 0
        for task_id in INCLUDED_TASKS:
            task = TASKS[task_id]
            directory = AREA / "results" / task.id / task.default_split
            present = {
                p.parent.name
                for p in directory.glob("*/metadata.json")
                if not p.parent.name.startswith(".")
            }
            self.assertTrue(expected.issubset(present), (task.id, present))
            for model in expected:
                with self.subTest(task=task.id, model=model):
                    frame, meta = load_bundle(task, directory / model)
                    self.assertEqual(meta["schema_version"], 5)
                    self.assertNotIn("effective_model", meta)
                    self.assertNotIn("unsupported_tracks", meta)
                    self.assertGreaterEqual(meta["runtime"]["max_context"], task.history_length)
                    self.assertEqual(
                        len(frame), len(select_cycles(task)) * task.horizon * len(task.tracks)
                    )
                    rows += len(frame)
                    calls += len(meta["timings"])
                    self.assertNotIn("baseline", frame.columns)
        self.assertEqual(rows, 33600)
        self.assertEqual(calls, 1720)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("chronos", sys.modules)
        self.assertNotIn("timestral", sys.modules)

    def test_rankings_use_matching_windows_and_reserved_dates_stay_unscored(self):
        for task in TASKS.values():
            with self.subTest(task=task.id):
                view = load_comparison(task)
                self.assertTrue(view.summary.Windows.eq(len(select_cycles(task))).all())
        task = TASKS["rte-hourly-24-v1"]
        view = load_comparison(task)
        self.assertEqual(set(view.predictions.timestamp.dt.year), {2024})
        self.assertFalse((AREA / "results" / task.id / "evaluation").exists())


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("evaluation", AREA / "evaluation.py")
        notebook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notebook)
        cls.app = notebook.app

    def test_default_view_is_a_small_weather_comparison(self):
        _, definitions = self.app.run()
        self.assertEqual(definitions["task"].dataset, "weather")
        self.assertEqual(definitions["track_picker"].value, "target_only")
        self.assertEqual(set(definitions["model_picker"].value), {"chronos2", "timestral-5m"})
        self.assertFalse(definitions["show_intervals"].value)
        self.assertEqual(len(definitions["summary"]), 4)

    def test_all_tasks_support_advanced_controls_without_changing_scores(self):
        for task in TASKS.values():
            with self.subTest(task=task.id):
                before = load_comparison(task, strict=True)
                models = list(before.provenance["Model ID"])
                _, definitions = self.app.run(defs={
                    "task_picker": SimpleNamespace(value=task.id),
                    "track_picker": SimpleNamespace(value="calendar"),
                    "model_picker": SimpleNamespace(value=models),
                    "show_intervals": SimpleNamespace(value=True),
                    "date_picker": SimpleNamespace(value=before.schedule.iloc[-1].forecast_start),
                    "case_picker": SimpleNamespace(value="failure"),
                })
                pd.testing.assert_frame_equal(definitions["comparison"].scores, before.scores)
                expected = before.summary[before.summary.Track == "calendar"].sort_values(
                    [task.primary_metric, "Model ID"])
                pd.testing.assert_frame_equal(definitions["summary"], expected)

    def test_missing_and_invalid_results_keep_a_working_baseline_view(self):
        task = TASKS["weather-daily-7-v1"]
        with tempfile.TemporaryDirectory() as directory:
            area = Path(directory)
            (area / "datasets").mkdir()
            for name in ("cycles.csv", "strasbourg_daily.csv"):
                shutil.copy(AREA / "datasets" / name, area / "datasets" / name)
            for corrupt in (False, True):
                if corrupt:
                    bundle = area / "results" / task.id / task.default_split / "broken"
                    bundle.mkdir(parents=True)
                    (bundle / "metadata.json").write_text('{"schema_version":4}')
                view = load_comparison(task, area)
                with patch("foundation_timeseries.results.load_comparison", return_value=view):
                    outputs, definitions = self.app.run()
                self.assertTrue(definitions["comparison"].predictions.empty)
                self.assertTrue(definitions["summary"]["Model ID"].str.startswith("baseline:").all())
                rendered = "".join(output.text for output in outputs if output is not None)
                self.assertIn("No valid model forecasts", rendered)
                if corrupt:
                    self.assertIn("Unavailable results", rendered)
                    self.assertIn("expected schema v5", rendered)


if __name__ == "__main__":
    unittest.main()
