# Datasets and experiment contract

Start with the shared evaluation notebook. Read the target and information boundary, inspect baselines, then compare saved forecasts. Entity/date controls change only the worked example; model selection filters displayed participants without changing any scores. The default view shows weather, target-only inputs, Chronos-2, Timestral-5M, and baselines. Expand optional sections for other tracks, checkpoints, uncertainty, diagnostics, and cost. Selecting an input track selects a distinct declared experiment over the same eligible windows.

| Task ID | Target | History | Horizon | Windows per track | Data card |
| --- | --- | ---: | ---: | ---: | --- |
| `weather-daily-7-v1` | Daily maximum temperature, °C | 730 days | 7 days | 12 exploratory | [Weather](weather.md) |
| `bikes-2hourly-12-v1` | Rentals per two-hour bin | 4,380 bins | 12 bins | 12 exploratory | [Bikes](bikes.md) |
| `retail-monthly-6-v1` | Monthly revenue, million USD | 120 months | 6 months | 4 exploratory | [Retail](retail.md) |
| `rte-hourly-24-v1` | Regional mean hourly demand, MW | 1,344 hours | 24 hours | 120 development | [RTE](rte.md) |

Previously inspected weather/bikes/retail dates were formerly labeled `evaluation`; they are now explicitly **exploratory**. The source CSVs, forecast dates, histories, and horizons are preserved. RTE retains 126 reserved 2025 windows; no command in the model exporters unlocks them. The shared schedule includes task IDs and remains readable CSV.

## Information tracks

`target_only` uses target history. `calendar` adds historical and future numeric calendar values. `past_observed` adds historical observed weather only. `past_observed_calendar` combines those two additions. Weather and bikes have all four; retail and RTE have only the first two. RTE's hour/weekday use Europe/Paris, while observations and forecast origins remain UTC. No encoding selection is performed on the outcomes.

Each model receives one entity at a time and the same complete declared context/horizon. Future targets and future observed weather never enter the adapter interface. Missing history stays distinct from zero. Bikes use native library handling; RTE requires complete historical and held-out targets. Later origins can use newly observed actuals from earlier windows.

## Saved results

Schema-v5 bundles live in `timeseries/results/<task>/<split>/<model>/`, with `predictions.csv` and `metadata.json`. Prediction CSVs identify entity, track, origin, timestamp, step, actual, point, and point interpretation; p10/p50/p90 are optional. Model IDs identify participants; names are display labels.

Metadata records a semantic `task_contract`, source/prediction hashes, the selected schedule, one static model specification, checkpoint/library/lockfile provenance, observed runtime limits and decoding settings, hardware, and per-window timings. `model` identifies the requested configuration; `runtime` records observations from the loaded checkpoint without duplicating that specification. Historical environment hashes describe the original runs, even after packaging changes. Labels and Python formatting do not affect compatibility. Shared input-behavior changes require an explicit `INPUT_VERSION` increment.

The schedule contains only `task`, `entity`, `forecast_start`, and `split`; task declarations supply lengths, frequency, and seasonal period. Several tasks can share a dataset. Preparation scripts replace only their own task IDs.

A model/task bundle is published only after all declared tracks succeed. The first inference or validation failure raises a contextual error and leaves previous results untouched. Staging validation and interruption recovery protect replacement; smoke outputs remain temporary and cannot enter comparisons. The included metadata was migrated from v4 to v5 offline without rewriting forecasts. Only v5 is supported. Context exceeding a loaded model's limit raises an error before inference; inputs are never silently truncated.

## Baselines and scores

Persistence repeats the last finite observation. Seasonal naïve repeats the latest historical pattern and falls back to earlier nonmissing values at the same phase. Four-week averages use available values among exactly four weekly references. Horizons longer than a season repeat historical phases without indexing test outcomes. A missing reference gives an undefined prediction, never a fabricated zero.

Weather uses annual lag 365; bikes weekly lag 84; retail annual lag 12; RTE has daily lag 24 and weekly lag 168. Seasonal naïve is the fixed reference for the first three tasks; weekly naïve is RTE's reference.

MAE is the main score for the three teaching tasks; RTE ranks by MASE. Signed bias is forecast minus actual. MASE divides window MAE by mean absolute seasonal differences in the permitted history, using finite historical pairs. No pairs or a zero scale make scaled scores undefined. Average origins within each entity, then entities equally; undefined values propagate.

Improvement is `100 × (1 − aggregate model score / aggregate reference score)` on matching windows, using the task's named primary metric. Zero reference scores produce undefined percentages. Related timestamps and regions are not independent trials.

For supported quantiles, the 80% interval score is `width + 10 × lower_miss + 10 × upper_miss`; WIS is `(0.5 × absolute_median_error + 0.1 × interval_score) / 1.5`. Lower is better. Also show scaled WIS, marginal 80% coverage, and interval width. Point-only models and deterministic baselines remain valid point participants. Coverage is not guaranteed, and daily quantiles cannot simply be summed into a valid total-uncertainty band.

## Reproduce

From the repository root:

```sh
# Included data/results only; no model libraries or weights.
uv run --project timeseries marimo edit timeseries/evaluation.py

# Check contracts without model dependencies.
uv run --project timeseries python -m unittest discover -s timeseries/tests -v

# Smoke every selected task/track before refreshing (temporary outputs).
uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py --smoke
uv run --project timeseries/timestral python timeseries/timestral/evaluate.py --size all --smoke

# Full refresh: Chronos-2 and all four Timestral sizes.
uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py --suite showcase --refresh
uv run --project timeseries/timestral python timeseries/timestral/evaluate.py --suite showcase --size all --refresh

# One task, or the original three teaching tasks; Timestral defaults to 5M.
uv run --project timeseries/timestral python timeseries/timestral/evaluate.py --task rte-hourly-24-v1
uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py --suite teaching

# Validate all available bundles and regenerate documentation from saved outputs.
uv run --project timeseries python -m foundation_timeseries.results --markdown timeseries/results/README.md

# Explicit source refresh, separate from inference and notebook viewing.
uv run --project timeseries python timeseries/datasets/prepare.py
uv run --project timeseries python timeseries/datasets/prepare_rte.py
```

`--task` is repeatable and mutually exclusive with `--suite`. The default suite is `showcase`; `teaching` is the explicit `TEACHING_TASKS` list. Matching complete bundles are skipped unless `--refresh` is supplied. Timestral accepts `5M`, `28M`, `123M`, `209M`, or `all`. CPU thread settings can be controlled through `OMP_NUM_THREADS`/`MKL_NUM_THREADS`; effective Torch threads and hardware are recorded. Inference timings exclude loading/downloads and are not controlled latency benchmarks.

The full declared refresh is 1,720 calls and 33,600 prediction rows across 20 task/model bundles, excluding smoke calls. [Generated tables](../results/README.md) report measured outcomes. To extend the showcase, follow the definitions and adapters described below.

## Extend the showcase

The shared package lives directly under `timeseries/foundation_timeseries/`; no `src/` directory or import-path setup is needed after uv installs the editable dependency.

For another dataset, add a CSV and a source card, declare its `DatasetSpec` and `TaskSpec` in `tasks.py`, and add fixed origins to `cycles.csv`. Keep the source and task separate: several forecasting questions can use the same dataset. Baselines are labelled callables such as `partial(seasonal_naive, lag=168)`, rather than numeric mode switches.

For another model, follow either family's small adapter: a static `ModelSpec`, observed `runtime` details including `max_context`, and `predict(inputs)` returning a `Prediction`. Each adapter must handle all declared task tracks or fail clearly. Inputs contain target history, past observations, known historical/future covariates, and the horizon; they never contain future target values. Quantiles may be omitted for a point-only model. Keep model-specific array conversion and the native API call together.

The runner follows **history → permitted inputs → prediction → validation → saved results**. `forecast_task` performs inference; `describe_run` records provenance; `write_bundle` stages and validates a complete result before replacing it. The shared metrics need no dataset-name branches. The notebook reads saved results and does not perform inference.
