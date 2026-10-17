import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from hashlib import sha256
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update({"font.size": 12})
    return Path, json, mdates, mo, np, pd, plt, sha256


@app.cell
def _(Path):
    area = Path(__file__).resolve().parent
    return (area,)


@app.cell
def _(mo):
    mo.md("""
    # Seasonal forecasting comparisons

    Chronos-2 and four Timestral sizes make repeated **zero-shot** forecasts without training.
    Each cycle uses observations available before its forecast start, including actuals observed since earlier cycles.
    **Covariates** add historical weather and calendar features known in advance.
    Viewing or selecting a cycle reads saved results; it never loads a model.

    **MAE** is mean absolute error in the target's units. **Bias** is mean forecast minus actual;
    positive means overforecasting. Weather ranks by MAE in °C, without a score.
    Demand ranks by **[Vandeput score](https://supchains.com/vn1-forecasting-competition/) (%) = MAE (%) + |Bias (%)|**.
    Within each cycle, both percentages divide error sums by total actual demand, then multiply by 100.
    Rankings average those per-cycle metrics equally, applying the absolute bias penalty before averaging.
    Smaller is better; the seasonal naïve baseline repeats an earlier season.
    Shading shows 10th–90th percentile forecasts, not guaranteed coverage.
    """)
    return


@app.cell
def _():
    # CSV, heading, target unit
    examples = {
        "weather": ("strasbourg_daily.csv", "Strasbourg weather · daily", "°C"),
        "bikes": ("bikes_2hourly.csv", "Bike demand · every two hours", "rentals / 2 hours"),
        "retail": ("retail_monthly.csv", "Retail sales · monthly", "million USD"),
    }
    model_names = {
        "chronos2": "Chronos-2",
        "timestral-5m": "Timestral-5M", "timestral-28m": "Timestral-28M",
        "timestral-123m": "Timestral-123M", "timestral-209m": "Timestral-209M",
    }
    colors = {
        "Seasonal naïve": "#666666", "Chronos-2": "#2364AA", "Timestral-5M": "#A07810",
        "Timestral-28M": "#5D6B31", "Timestral-123M": "#9A3D70", "Timestral-209M": "#C15F14",
    }
    styles = {
        "Seasonal naïve": ":", "Chronos-2": "--", "Timestral-5M": ":",
        "Timestral-28M": "-.", "Timestral-123M": (0, (5, 1)), "Timestral-209M": "-",
    }
    modes = ["Target only", "With covariates"]
    return colors, examples, model_names, modes, styles


@app.cell
def _(area, examples, json, model_names, modes, np, pd, sha256):
    # Validate the schedule and sources before accepting any saved predictions.
    issues, datasets, windows = [], {}, {}
    cycles = pd.DataFrame()
    _frames, _timings, _provenance = [], [], []
    try:
        _schedule_path = area / "datasets" / "cycles.csv"
        _schedule_hash = sha256(_schedule_path.read_bytes()).hexdigest()
        _schedule_records = pd.read_csv(_schedule_path).to_dict("records")
        cycles = pd.read_csv(_schedule_path, parse_dates=["forecast_start"])
        assert set(cycles.columns) == {"dataset", "forecast_start", "history_length", "horizon", "seasonal_lag", "frequency"}
        assert set(cycles.dataset) == set(examples) and not cycles.duplicated(["dataset", "forecast_start"]).any()
        assert (cycles[["history_length", "horizon", "seasonal_lag"]] > 0).all().all()
        for _id, (_file, _title, _unit) in examples.items():
            _data = pd.read_csv(area / "datasets" / _file, parse_dates=["timestamp"]).set_index("timestamp", drop=False)
            assert _data.index.is_unique and _data.index.is_monotonic_increasing
            datasets[_id] = _data
            for _cycle in cycles[cycles.dataset == _id].itertuples(index=False):
                _position = _data.index.get_loc(_cycle.forecast_start)
                _history = _data.iloc[_position - _cycle.history_length:_position]
                _future = _data.iloc[_position:_position + _cycle.horizon]
                assert len(_history) == _cycle.history_length and len(_future) == _cycle.horizon
                assert _future.index.equals(pd.date_range(_cycle.forecast_start, periods=_cycle.horizon, freq=_cycle.frequency))
                assert _history.index.equals(pd.date_range(end=_cycle.forecast_start, periods=_cycle.history_length + 1, freq=_cycle.frequency)[:-1])
                assert np.isfinite(_future.drop(columns="timestamp").to_numpy()).all(), "incomplete evaluation targets or covariates"
                _baseline = []
                for _step in range(_cycle.horizon):
                    _earlier = _history.target.iloc[_step - _cycle.seasonal_lag::-_cycle.seasonal_lag].dropna()
                    assert not _earlier.empty, "no historical seasonal baseline"
                    _baseline.append(_earlier.iloc[0])
                windows[(_id, _cycle.forecast_start)] = (_history, _future, np.array(_baseline))
    except (OSError, ValueError, KeyError, TypeError, AssertionError, IndexError) as _error:
        issues.append(f"Data/schedule: {str(_error) or 'invalid windows'}")

    if not issues:
        for _key, _name in model_names.items():
            try:
                _path = area / "results" / f"{_key}.csv"
                _meta = json.loads(_path.with_suffix(".json").read_text())
                assert _meta.get("schema_version") == 2, "old result format"
                assert _meta["cycles_sha256"] == _schedule_hash and _meta["cycles"] == _schedule_records, "stale cycle schedule"
                assert sha256(_path.read_bytes()).hexdigest() == _meta["predictions_sha256"], "prediction file changed"
                _frame = pd.read_csv(_path, parse_dates=["forecast_start", "timestamp"])
                assert set(_frame.columns) == {"dataset", "mode", "forecast_start", "timestamp", "step", "actual", "baseline", "p10", "p50", "p90"}
                assert not _frame.duplicated(["dataset", "mode", "forecast_start", "step"]).any(), "duplicate predictions"
                assert set(zip(_frame.dataset, _frame["mode"], _frame.forecast_start)) == {
                    (_cycle.dataset, _mode, _cycle.forecast_start)
                    for _cycle in cycles.itertuples(index=False) for _mode in modes
                }, "incomplete scenarios"
                assert np.isfinite(_frame[["actual", "baseline", "p10", "p50", "p90"]].to_numpy()).all()
                assert (_frame.p10 <= _frame.p50).all() and (_frame.p50 <= _frame.p90).all()
                _model_timings = []
                for _id, (_file, _title, _unit) in examples.items():
                    assert sha256((area / "datasets" / _file).read_bytes()).hexdigest() == _meta["dataset_sha256"][_id], f"stale {_id} data"
                for _cycle in cycles.itertuples(index=False):
                    _history, _future, _baseline = windows[(_cycle.dataset, _cycle.forecast_start)]
                    for _mode in modes:
                        _group = _frame[(_frame.dataset == _cycle.dataset) & (_frame["mode"] == _mode) & (_frame.forecast_start == _cycle.forecast_start)]
                        assert len(_group) == _cycle.horizon, "incorrect forecast length"
                        assert np.array_equal(_group.step, np.arange(1, _cycle.horizon + 1))
                        assert np.array_equal(_group.timestamp.to_numpy(), _future.timestamp.to_numpy())
                        assert np.allclose(_group.actual, _future.target, rtol=0, atol=1e-9), "actuals differ from source"
                        assert np.allclose(_group.baseline, _baseline, rtol=0, atol=1e-9), "baseline differs from historical values"
                        _seconds = _meta["inference_seconds"][_cycle.dataset][_mode][str(_cycle.forecast_start)]
                        assert np.isfinite(_seconds) and _seconds >= 0
                        _model_timings.append({"dataset": _cycle.dataset, "Model": _name, "Mode": _mode, "forecast_start": _cycle.forecast_start, "Seconds": _seconds})
                _frames.append(_frame.assign(Model=_name))
                _timings.extend(_model_timings)
                _provenance.append({"Model": _name, "Checkpoint": _meta["revision"], "Library": _meta["library_version"], "Generated (UTC)": _meta["generated_at"]})
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as _error:
                issues.append(f"{_name}: {str(_error) or 'inconsistent results'}")
    predictions = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    timings = pd.DataFrame(_timings)
    provenance = pd.DataFrame(_provenance)
    return cycles, datasets, issues, predictions, provenance, timings, windows


@app.cell
def _(issues, mo):
    mo.stop(bool(issues), mo.callout(mo.md(
        "**Results need refreshing.** " + "; ".join(issues)
        + "\n\nFrom the repository root, run:\n\n```sh\n"
        + "uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py\n"
        + "uv run --project timeseries/timestral python timeseries/timestral/evaluate.py\n```\n"
        + "If data or `cycles.csv` is missing, first run `uv run --project timeseries/chronos2 python timeseries/datasets/prepare.py`."
    ), kind="warn"))
    ready = True
    return (ready,)


@app.cell
def _(examples, np, pd, predictions, ready):
    assert ready

    def measure(actual, forecast, dataset):
        errors = forecast - actual
        values = {"MAE": errors.abs().mean(), "Bias": errors.mean()}
        if dataset != "weather":
            valid = (actual >= 0).all() and actual.sum() > 0
            mae_pct = 100 * errors.abs().sum() / actual.sum() if valid else np.nan
            bias_pct = 100 * errors.sum() / actual.sum() if valid else np.nan
            values.update({"MAE (%)": mae_pct, "Bias (%)": bias_pct, "Score (%)": mae_pct + abs(bias_pct)})
        return values

    _rows = []
    for (_id, _start), _selected in predictions.groupby(["dataset", "forecast_start"], sort=False):
        _reference = _selected.drop_duplicates("timestamp")
        _rows.append({"dataset": _id, "forecast_start": _start, "Model": "Seasonal naïve", "Mode": "Both", **measure(_reference.actual, _reference.baseline, _id)})
        for (_name, _mode), _group in _selected.groupby(["Model", "mode"], sort=False):
            _rows.append({"dataset": _id, "forecast_start": _start, "Model": _name, "Mode": _mode, **measure(_group.actual, _group.p50, _id)})
    cycle_metrics = pd.DataFrame(_rows)
    _groups = cycle_metrics.groupby(["dataset", "Model", "Mode"], sort=False)
    _means = _groups[["MAE", "Bias", "MAE (%)", "Bias (%)", "Score (%)"]].agg(lambda values: values.mean(skipna=False))
    _means["Cycles"] = _groups.size()
    metrics = _means.reset_index()
    rankings = {}
    _winners = []
    for _id, (_file, _title, _unit) in examples.items():
        _metric = "MAE" if _id == "weather" else "Score (%)"
        _ranking = metrics[metrics.dataset == _id].copy()
        _ranking["Rank"] = _ranking[_metric].rank(method="min").astype("Int64")
        _ranking = _ranking.sort_values([_metric, "Model", "Mode"], na_position="last").reset_index(drop=True)
        rankings[_id] = _ranking
        for _record in _ranking[_ranking.Rank == 1].itertuples(index=False):
            _value = _ranking.loc[(_ranking.Model == _record.Model) & (_ranking.Mode == _record.Mode), _metric].iloc[0]
            _winners.append({"Dataset": _title.split(" · ")[0], "Cycles": _record.Cycles, "Best result": f"{_record.Model} · {_record.Mode.lower()}", "Metric": "Mean cycle MAE" if _id == "weather" else "Mean cycle score", "Value": _value, "Unit": _unit if _id == "weather" else "%"})
    winners = pd.DataFrame(_winners)
    return cycle_metrics, measure, metrics, rankings, winners


@app.cell
def _(colors, cycles, datasets, examples, mdates, mo, modes, pd, plt, rankings, ready, cycle_metrics, styles, timings):
    assert ready

    def date_axis(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)

    def html_table(frame, decimals=3):
        return mo.Html('<div style="overflow-x: auto">' + frame.round(decimals).to_html(index=False, border=0, na_rep="—") + '</div>')

    def dataset_summary(dataset):
        filename, title, unit = examples[dataset]
        schedule = cycles[cycles.dataset == dataset]
        first, last = schedule.iloc[0], schedule.iloc[-1]
        offset = pd.tseries.frequencies.to_offset(last.frequency)
        end = last.forecast_start + last.horizon * offset
        beginning = pd.Timestamp(year=first.forecast_start.year, month=1, day=1)
        actual = datasets[dataset].loc[beginning:end - offset, "target"]
        overview_unit = unit
        if dataset == "bikes":
            actual = actual.resample("D").sum(min_count=12)
            overview_unit = "rentals / day · complete days only"
        blocks = [mo.md(f"""
        ## {title}
        **Cycles:** {len(schedule)} · **History per forecast:** {first.history_length} observations · **Horizon:** {first.horizon} steps

        **Evaluation period:** {beginning:%Y-%m-%d} → {end:%Y-%m-%d} (end exclusive) · **Forecast unit:** {unit}
        """)]
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        ax.plot(actual.index, actual, color="#555555", linewidth=1.2, label="Actual")
        for number, cycle in enumerate(schedule.itertuples(index=False)):
            ax.axvspan(cycle.forecast_start, cycle.forecast_start + cycle.horizon * offset, color="#2364AA", alpha=0.10, label="Forecast windows" if number == 0 else None)
            ax.axvline(cycle.forecast_start, color="#2364AA", alpha=0.25, linewidth=0.8)
        ax.set(title="Evaluation period overview", ylabel=overview_unit)
        date_axis(ax)
        ax.legend(fontsize=10)
        blocks.append(fig)
        plt.close(fig)
        if dataset == "bikes":
            blocks.append(mo.md("The overview sums complete days; detailed forecasts use two-hour totals. Missing hours remain gaps, never zero rentals. The 4,380-step context spans 365 days and fits every checkpoint."))

        ranking = rankings[dataset]
        metric = "MAE" if dataset == "weather" else "Score (%)"
        label = f"Mean cycle MAE ({unit})" if dataset == "weather" else "Mean cycle Vandeput score (%)"
        columns = ["Rank", "Model", "Mode", "MAE", "Bias", "Cycles"] if dataset == "weather" else ["Rank", "Model", "Mode", "Score (%)", "Cycles", "MAE", "MAE (%)", "Bias (%)"]
        blocks.extend([
            mo.md("### Ranked results\nMeans of per-cycle metrics, with every cycle weighted equally. Lower is better. Exact ties share a rank; undefined metrics sort last and are never omitted from an average."),
            html_table(ranking[columns]),
        ])
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        for row, record in ranking.iterrows():
            value = record[metric]
            ax.barh(f"{record.Model} · {record.Mode.lower()}", value, color=colors[record.Model], alpha=0.85)
            if pd.notna(value):
                ax.annotate(f" {value:,.2f}", (value, row), va="center", fontsize=11)
        maximum = ranking[metric].max()
        ax.set(xlabel=label, xlim=(0, maximum * 1.22 if pd.notna(maximum) and maximum > 0 else 1))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        blocks.append(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True, layout="constrained")
        values = cycle_metrics[cycle_metrics.dataset == dataset]
        for ax, mode in zip(axes, modes):
            for name, group in values[(values.Mode == mode) | (values.Mode == "Both")].groupby("Model", sort=False):
                group = group.sort_values("forecast_start")
                ax.plot(group.forecast_start, group[metric], linestyle=styles[name], marker="o", markersize=3, color=colors[name], label=name)
            ax.set(title=mode, ylabel="MAE (°C)" if dataset == "weather" else "Vandeput score (%)", ylim=(0, None))
            date_axis(ax)
        axes[-1].set_xlabel("Forecast start")
        starts = schedule.forecast_start.tolist()
        axes[-1].set_xticks(starts if len(starts) <= 6 else [*starts[::3], starts[-1]])
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=10, ncol=3, loc="outside lower center")
        blocks.extend([mo.md("### Performance by cycle\nEach point scores one complete forecast horizon."), fig])
        plt.close(fig)
        selected_timings = timings[timings.dataset == dataset].groupby(["Model", "Mode"], sort=False).Seconds.agg(**{"Cycles": "size", "Mean (s)": "mean", "Total (s)": "sum"}).reset_index()
        blocks.extend([
            mo.md("### Inference time\nSingle CPU call per cycle and mode; excludes loading and downloads, without warm-up or repeated timing trials."),
            html_table(selected_timings),
            mo.md("### Inspect a forecast cycle\nThe selection below changes only the detailed plots; rankings above always use all cycles."),
        ])
        return mo.vstack(blocks)

    def cycle_picker(dataset):
        options = cycles.loc[cycles.dataset == dataset, "forecast_start"].dt.strftime("%Y-%m-%d").tolist()
        return mo.ui.dropdown(options=options, value=options[-1], label="Forecast start", allow_select_none=False)

    return cycle_picker, dataset_summary, date_axis, html_table


@app.cell
def _(colors, date_axis, examples, mo, modes, pd, plt, predictions, ready, styles, windows):
    assert ready

    def detailed_forecasts(dataset, selected):
        start = pd.Timestamp(selected)
        history, future, baseline = windows[(dataset, start)]
        unit = examples[dataset][2]
        blocks = [mo.md(f"**Forecast timestamps:** {future.timestamp.iloc[0]:%Y-%m-%d %H:%M} → {future.timestamp.iloc[-1]:%Y-%m-%d %H:%M} · **Unit:** {unit}")]
        if dataset == "bikes":
            blocks.append(mo.md("Each timestamp labels the start of a two-hour bin; the final bin ends at midnight."))
        for mode in modes:
            displayed = predictions[(predictions.dataset == dataset) & (predictions["mode"] == mode) & (predictions.forecast_start == start)]
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            recent = history.tail(3 * len(future))
            ax.plot(recent.timestamp, recent.target, color="#666666", label="History")
            ax.plot(future.timestamp, future.target, "o-", color="#111111", label="Actual", markersize=3)
            ax.plot(future.timestamp, baseline, ":", color=colors["Seasonal naïve"], label="Seasonal naïve", linewidth=2)
            for name, group in displayed.groupby("Model", sort=False):
                ax.plot(group.timestamp, group.p50, linestyle=styles[name], color=colors[name], label=name, linewidth=2)
                ax.fill_between(group.timestamp, group.p10, group.p90, color=colors[name], alpha=0.08)
            ax.set(title=mode, ylabel=unit, xlabel="Date / source local time")
            date_axis(ax)
            ax.legend(fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            blocks.append(fig)
            plt.close(fig)
        return mo.vstack(blocks)

    return (detailed_forecasts,)


@app.cell
def _(dataset_summary):
    dataset_summary("weather")
    return


@app.cell
def _(cycle_picker):
    weather_cycle = cycle_picker("weather")
    weather_cycle
    return (weather_cycle,)


@app.cell
def _(detailed_forecasts, weather_cycle):
    detailed_forecasts("weather", weather_cycle.value)
    return


@app.cell
def _(dataset_summary):
    dataset_summary("bikes")
    return


@app.cell
def _(cycle_picker):
    bike_cycle = cycle_picker("bikes")
    bike_cycle
    return (bike_cycle,)


@app.cell
def _(bike_cycle, detailed_forecasts):
    detailed_forecasts("bikes", bike_cycle.value)
    return


@app.cell
def _(dataset_summary):
    dataset_summary("retail")
    return


@app.cell
def _(cycle_picker):
    retail_cycle = cycle_picker("retail")
    retail_cycle
    return (retail_cycle,)


@app.cell
def _(detailed_forecasts, retail_cycle):
    detailed_forecasts("retail", retail_cycle.value)
    return


@app.cell
def _(html_table, mo, winners):
    mo.vstack([
        mo.md("""
        ## Recap by dataset
        Best results use mean cycle MAE for weather and mean cycle Vandeput score for demand.
        Each dataset stands on its own: no cross-dataset average or overall ranking.
        """),
        html_table(winners),
    ])
    return


@app.cell
def _(html_table, mo, provenance):
    mo.accordion({"Result provenance": html_table(provenance)})
    return


@app.cell
def _(mo):
    mo.md("""
    These seasonally distributed cycles demonstrate behavior, not general model quality.
    Public data may overlap pretraining. Strasbourg weather is ERA5 reanalysis; retail revenue is a demand proxy.
    Future observed weather is excluded from inference. Sources, windows, and refresh commands: `timeseries/datasets/README.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
