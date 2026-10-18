# Foundation models

Small, readable experiments with pretrained forecasting models: **Chronos-2** and **Timestral**. Learn the native APIs, then ask whether their forecasts improve on simple baselines.

**Zero-shot** means forecasting without task-specific training here. These public datasets may have appeared in pretraining; the comparisons do not establish otherwise.

With [uv](https://docs.astral.sh/uv/) installed, run these commands from the repository root.

## 1. Make your first forecast

A model reads a **context** of observed values and predicts a **horizon** of future steps. Each notebook shows the model call directly, followed by a plot against the actual continuation and a persistence baseline.

```sh
uv run --project timeseries/chronos2 marimo edit timeseries/chronos2/quick_start.py
uv run --project timeseries/timestral marimo edit timeseries/timestral/quick_start.py
```

Examples use CPU and download pinned weights on first use. Start with Timestral-5M; change `size` to explore 28M, 123M, or 209M.

## 2. Add useful information

**Covariates** are extra signals. Sensor observations are available only in the past; a planned schedule or calendar is known in the future. The next notebooks make that distinction visible in the model inputs.

```sh
uv run --project timeseries/chronos2 marimo edit timeseries/chronos2/covariates.py
uv run --project timeseries/timestral marimo edit timeseries/timestral/covariates.py
```

These synthetic examples explain the API. Additional inputs do not guarantee better predictions.

## 3. Compare with a simple baseline

Explore the included forecasts **without model libraries, downloads, or weights**:

```sh
uv run --project timeseries marimo edit timeseries/evaluation.py
```

Start with daily temperature, two models, a plot, and an accuracy table. Expand the optional sections to compare checkpoints, covariates, uncertainty, difficult cases, and inference time. A **quantile** describes a forecast percentile; the 10th–90th percentiles form a native 80% interval, whose observed coverage can differ from 80%.

For example, on the 12 exploratory weather windows with target history only, Timestral-5M has **MAE 2.760 °C**, persistence **3.248 °C**, and seasonal naïve **4.435 °C**. MAE is average absolute error; lower is better. That improvement is specific to this task and schedule. See the [saved comparisons](timeseries/results/README.md) for all results.

## Find your way around

- `timeseries/chronos2/` and `timeseries/timestral/`: direct API lessons and a small evaluation adapter for each family, with separate locked environments.
- `timeseries/foundation_timeseries/`: shared task definitions, baselines, metrics, result validation, and execution.
- `timeseries/datasets/`: frozen data, source cards, and explicit preparation scripts.
- `timeseries/results/`: saved predictions and reproducibility metadata.
- `timeseries/tests/`: lightweight behavioral checks; no model weights required.

The [dataset guide](timeseries/datasets/README.md) explains all four tasks, methodology, extension points, and reproduction commands. Model notes cover [Chronos-2](timeseries/chronos2/README.md) and [Timestral](timeseries/timestral/README.md). The comparison includes daily temperature, two-hour bike demand, monthly retail revenue, and hourly regional electricity. **RTE's 2025 holdout remains reserved, unscored, and unplotted.**
