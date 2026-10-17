# Foundation models

Readable foundation model experiments. A pretrained model has learned from other data; zero-shot forecasting applies it to a new series without additional training.

Each area has its own folder. Each model has an independent uv environment, Python version, and lockfile. The first area, **time series**, includes **Chronos-2** and **Timestral-209M**.

Start here:

1. **Understand one model:** its `quick_start.py` forecasts a simple synthetic series in six cells.
2. **Explore extra information:** its `covariates.py` demonstrates past and known-future covariates with direct model calls.
3. **Compare models:** the shared evaluation notebook shows daily Strasbourg weather, hourly bike demand, and monthly retail sales in separate sections, followed by a recap.

Install [uv](https://docs.astral.sh/uv/), then launch from the repository root:

```sh
uv run --project timeseries/chronos2 marimo edit timeseries/chronos2/quick_start.py
uv run --project timeseries/timestral marimo edit timeseries/timestral/quick_start.py
uv run --project timeseries marimo edit timeseries/evaluation.py
```

For covariates, substitute `covariates.py` for `quick_start.py` in either model command. Teaching notebooks use CPU inference and download pinned checkpoints on first use. Evaluation reads included results without model libraries or weights.

See [datasets and result-refresh commands](timeseries/datasets/README.md). These illustrative holdouts are not a leaderboard. Future areas such as `decisions/` follow the same layout. Keep code minimal; fine-tuning and Azure ML remain deferred.
