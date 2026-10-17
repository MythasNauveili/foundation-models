# Foundation models

Readable foundation model experiments. A pretrained model has learned from other data; zero-shot forecasting applies it to a new series without additional training.

Each area has a folder; each model family has its own uv environment and lockfile. **Time series** includes **Chronos-2** and **Timestral (5M, 28M, 123M, 209M)**.

Start here:

1. **Understand one model:** its `quick_start.py` forecasts a simple synthetic series in six cells.
2. **Explore extra information:** its `covariates.py` demonstrates past and known-future covariates with direct model calls.
3. **Compare models:** the shared notebook evaluates seasonal forecasts for daily Strasbourg weather, two-hour bike demand, and monthly retail sales, with separate rankings and cycle selectors.

Install [uv](https://docs.astral.sh/uv/), then launch from the repository root:

```sh
uv run --project timeseries/chronos2 marimo edit timeseries/chronos2/quick_start.py
uv run --project timeseries/timestral marimo edit timeseries/timestral/quick_start.py
uv run --project timeseries marimo edit timeseries/evaluation.py
```

For covariates, substitute `covariates.py` for `quick_start.py`. Timestral defaults to 5M; edit `size` to switch. Teaching runs on CPU and downloads pinned checkpoints on first use. Evaluation reads included results without model libraries or weights.

See [datasets and result-refresh commands](timeseries/datasets/README.md). Rankings apply within each dataset; there is no overall score. Future areas such as `decisions/` follow the same layout. Keep code minimal; fine-tuning and Azure ML remain deferred.
