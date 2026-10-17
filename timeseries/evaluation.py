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

    return Path, json, mdates, mo, np, pd, plt, sha256


@app.cell
def _(Path):
    area = Path(__file__).resolve().parent
    return (area,)


@app.cell
def _(mo):
    mo.md("""
    # Compare time-series foundation models

    Both pretrained models forecast the same unseen window without fine-tuning.
    These are saved CPU results: viewing this notebook downloads no model weights.
    **Covariates** are extra signals: past weather and calendar features known in advance.
    Each dataset below shows both scenarios, followed by a recap across the three examples.

    The **seasonal naïve** baseline repeats values from an earlier season.
    **MAE** is the mean absolute forecast error in the target's units; smaller is better.
    **Bias** is the mean forecast minus actual: positive means overforecasting.
    For demand, **[Vandeput score](https://supchains.com/vn1-forecasting-competition/) (%) = MAE (%) + |Bias (%)|**;
    smaller is better. MAE (%) divides the sum of absolute errors by total actual demand;
    Bias (%) divides the sum of signed errors by that same total, both multiplied by 100.
    This is different from averaging individual percentage errors (MAPE).
    Shading shows **10th–90th percentile forecasts**, not a guarantee of coverage.
    """)
    return


@app.cell
def _():
    # filename, frequency, horizon, seasonal lag, unit
    examples = {
        "Strasbourg weather · daily": ("weather", "strasbourg_daily.csv", "D", 7, 365, "°C"),
        "Bike demand · hourly": ("bikes", "bikes_hourly.csv", "h", 24, 168, "rentals"),
        "Retail sales · monthly": ("retail", "retail_monthly.csv", "MS", 6, 12, "million USD"),
    }
    model_names = {"chronos2": "Chronos-2", "timestral": "Timestral-209M"}
    colors = {"Chronos-2": "#2364AA", "Timestral-209M": "#C15F14"}
    modes = ["Target only", "With covariates"]
    return colors, examples, model_names, modes


@app.cell
def _(area, examples, json, model_names, modes, np, pd, sha256):
    # Keep invalid or stale results out of comparisons; regenerate with the commands below.
    issues, _frames, _timings, _provenance = [], [], [], []
    for _key, _name in model_names.items():
        try:
            _path = area / "results" / f"{_key}.csv"
            _meta = json.loads((area / "results" / f"{_key}.json").read_text())
            assert sha256(_path.read_bytes()).hexdigest() == _meta["predictions_sha256"], "prediction file changed"
            _frame = pd.read_csv(_path, parse_dates=["timestamp"])
            assert set(_frame.columns) == {"dataset", "mode", "timestamp", "actual", "baseline", "p10", "p50", "p90"}
            assert set(zip(_frame.dataset, _frame["mode"])) == {
                (_spec[0], _mode) for _spec in examples.values() for _mode in modes
            }, "incomplete scenarios"
            for _id, _file, _freq, _horizon, _lag, _unit in examples.values():
                _source = area / "datasets" / _file
                assert sha256(_source.read_bytes()).hexdigest() == _meta["dataset_sha256"][_id], f"stale {_id} data"
                for _mode in modes:
                    _group = _frame[(_frame.dataset == _id) & (_frame["mode"] == _mode)]
                    assert len(_group) == _horizon, "incorrect forecast length"
                    _seconds = _meta["inference_seconds"][_id][_mode]
                    assert np.isfinite(_seconds) and _seconds >= 0
                    _timings.append({"Dataset": _id, "Model": _name, "Mode": _mode, "Seconds": _seconds})
            _frames.append(_frame.assign(model=_name))
            _provenance.append({
                "Model": _name, "Checkpoint": _meta["revision"],
                "Library": _meta["library_version"], "Generated (UTC)": _meta["generated_at"],
            })
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as _error:
            issues.append(f"{_name}: {str(_error) or 'incomplete or inconsistent results'}")
    predictions = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    timings = pd.DataFrame(_timings)
    provenance = pd.DataFrame(_provenance)
    return issues, predictions, provenance, timings


@app.cell
def _(issues, mo):
    mo.stop(bool(issues), mo.callout(mo.md(
        "**Results need refreshing.** " + "; ".join(issues)
        + "\n\nFrom the repository root, run:\n\n```sh\n"
        + "uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py\n"
        + "uv run --project timeseries/timestral python timeseries/timestral/evaluate.py\n```"
    ), kind="warn"))
    ready = True
    return (ready,)


@app.cell
def _(examples, np, pd, predictions, ready):
    assert ready
    _metrics = []
    for _id, _file, _freq, _horizon, _lag, _unit in examples.values():
        _selected = predictions[predictions.dataset == _id]
        # The baseline is identical across models and modes; count each timestamp once.
        _reference = _selected.drop_duplicates("timestamp")
        _forecasts = [("Seasonal naïve", "Both", _reference.actual, _reference.baseline)]
        _forecasts.extend(
            (_name, _mode, _rows.actual, _rows.p50)
            for (_name, _mode), _rows in _selected.groupby(["model", "mode"], sort=False)
        )
        for _name, _mode, _actual, _forecast in _forecasts:
            _errors = _forecast - _actual
            _mae_pct, _bias_pct = np.nan, np.nan
            # Celsius has an arbitrary zero; percentages also need positive total demand.
            if _id != "weather" and (_actual >= 0).all() and _actual.sum() > 0:
                _mae_pct = 100 * _errors.abs().sum() / _actual.sum()
                _bias_pct = 100 * _errors.sum() / _actual.sum()
            _metrics.append({
                "Dataset": _id, "Model": _name, "Mode": _mode,
                "MAE": _errors.abs().mean(), "Bias": _errors.mean(),
                "MAE (%)": _mae_pct, "Bias (%)": _bias_pct,
                "Score (%)": _mae_pct + abs(_bias_pct),
            })
    metrics = pd.DataFrame(_metrics)
    return (metrics,)


@app.cell
def _(
    area,
    colors,
    examples,
    mdates,
    metrics,
    mo,
    modes,
    pd,
    plt,
    predictions,
    ready,
    timings,
):
    assert ready

    def dataset_section(title):
        dataset_id, filename, frequency, horizon, seasonal_lag, unit = examples[title]
        source = pd.read_csv(area / "datasets" / filename, parse_dates=["timestamp"])
        history = source.iloc[:-horizon]
        selected = predictions[predictions.dataset == dataset_id]
        blocks = [mo.md(f"""
        ## {title}
        **History:** {len(history)} observations · **Horizon:** {horizon} steps · **Unit:** {unit}  
        **Forecast window:** {source.timestamp.iloc[-horizon]:%Y-%m-%d %H:%M} → {source.timestamp.iloc[-1]:%Y-%m-%d %H:%M}
        """)]
        for mode in modes:
            displayed = selected[selected["mode"] == mode]
            reference = displayed[displayed.model == "Chronos-2"]
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            recent = history.tail(3 * horizon)
            ax.plot(recent.timestamp, recent.target, color="#666666", label="History")
            ax.plot(reference.timestamp, reference.actual, "o-", color="#111111", label="Actual", markersize=3)
            ax.plot(reference.timestamp, reference.baseline, ":", color="#777777", label="Seasonal naïve", linewidth=2)
            for name, group in displayed.groupby("model", sort=False):
                style = "--" if name == "Chronos-2" else "-"
                ax.plot(group.timestamp, group.p50, style, color=colors[name], label=name, linewidth=2)
                ax.fill_between(group.timestamp, group.p10, group.p90, color=colors[name], alpha=0.12)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set(title=mode, ylabel=unit, xlabel="Date / source local time")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            ax.legend(fontsize=9, ncol=2)
            blocks.append(fig)
            plt.close(fig)

        dataset_metrics = metrics[metrics.Dataset == dataset_id]
        model_metrics = dataset_metrics[dataset_metrics.Model != "Seasonal naïve"].reset_index(drop=True)
        baseline_mae = dataset_metrics.loc[dataset_metrics.Model == "Seasonal naïve", "MAE"].iloc[0]
        blocks.append(mo.md("### Does adding covariates help?"))
        fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        for row, record in model_metrics.iterrows():
            ax.barh(f"{record.Model} · {record.Mode.lower()}", record.MAE, color=colors[record.Model], alpha=0.85)
            ax.annotate(f" {record.MAE:,.2f}", (record.MAE, row), va="center", fontsize=9)
        ax.axvline(baseline_mae, color="#555555", linestyle=":", label=f"Seasonal naïve: {baseline_mae:,.2f}")
        ax.set(xlabel=f"MAE ({unit}) · smaller is better", xlim=(0, max(model_metrics.MAE.max(), baseline_mae) * 1.22))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)
        blocks.append(fig)
        plt.close(fig)
        if dataset_id == "weather":
            columns = ["Model", "Mode", "MAE", "Bias"]
            explanation = (
                "### Error and bias\nMAE and bias are in °C. Percentage scores are omitted because "
                "Celsius has an arbitrary zero ([why this matters](https://otexts.com/fpp3/accuracy.html))."
            )
        else:
            columns = ["Model", "Mode", "MAE (%)", "Bias (%)", "Score (%)"]
            explanation = "### Vandeput score\nAll percentages use this holdout's total actual demand. Lower score is better; positive bias means overforecasting."
        blocks.extend([
            mo.md(explanation),
            mo.Html('<div style="overflow-x: auto">' + dataset_metrics[columns].round(2).to_html(index=False, border=0, na_rep="—") + '</div>'),
        ])
        selected_timings = timings[timings.Dataset == dataset_id].drop(columns="Dataset")
        blocks.extend([
            mo.md("### Inference time\nSingle CPU calls, excluding model loading and downloads; no warm-up or repeated timing trials."),
            mo.Html(selected_timings.round(3).to_html(index=False, border=0)),
        ])
        return mo.vstack(blocks)

    return (dataset_section,)


@app.cell
def _(dataset_section):
    dataset_section("Strasbourg weather · daily")
    return


@app.cell
def _(dataset_section):
    dataset_section("Bike demand · hourly")
    return


@app.cell
def _(dataset_section):
    dataset_section("Retail sales · monthly")
    return


@app.cell
def _(examples, metrics, pd, ready):
    assert ready
    _best, _effects = [], []
    for _title, (_id, _file, _freq, _horizon, _lag, _unit) in examples.items():
        _selected = metrics[metrics.Dataset == _id]
        _baseline_mae = _selected.loc[_selected.Model == "Seasonal naïve", "MAE"].iloc[0]
        _candidates = {"Seasonal naïve": _baseline_mae}
        for _name, _group in _selected[_selected.Model != "Seasonal naïve"].groupby("Model", sort=False):
            _scores = dict(zip(_group.Mode, _group.MAE))
            _candidates.update({f"{_name} · {_mode.lower()}": _mae for _mode, _mae in _scores.items()})
            _without, _with = _scores["Target only"], _scores["With covariates"]
            _effects.append({
                "Dataset": _title.split(" · ")[0], "Model": _name,
                "MAE without": _without, "MAE with": _with,
                "Improvement (%)": 100 * (_without - _with) / _without if _without else float("nan"),
            })
        _winner = min(_candidates, key=_candidates.get)
        _best.append({"Dataset": _title.split(" · ")[0], "Lowest MAE forecast": _winner, "MAE": _candidates[_winner], "Unit": _unit})
    best_results = pd.DataFrame(_best)
    covariate_effects = pd.DataFrame(_effects)
    score_recap = metrics[metrics.Dataset != "weather"].pivot(
        index=["Model", "Mode"], columns="Dataset", values="Score (%)"
    ).rename(columns={"bikes": "Bike demand (%)", "retail": "Retail sales (%)"}).reset_index()
    score_recap.columns.name = None
    return best_results, covariate_effects, score_recap


@app.cell
def _(best_results, covariate_effects, mo, score_recap):
    mo.vstack([
        mo.md("""
        ## Recap
        These are the lowest MAEs on each fixed holdout, including the seasonal baseline.
        **Do not average raw MAE across datasets:** their units and scales differ.
        """),
        mo.Html('<div style="overflow-x: auto">' + best_results.round(3).to_html(index=False, border=0) + '</div>'),
        mo.md("""
        ### Vandeput score (%)
        Lower is better. This combines absolute error and absolute bias for the two demand examples;
        weather is excluded. Scores remain separate for each holdout; there is no overall ranking.
        """),
        mo.Html('<div style="overflow-x: auto">' + score_recap.round(2).to_html(index=False, border=0, na_rep="—") + '</div>'),
        mo.md("""
        ### Effect of covariates
        Positive improvement means lower error with covariates; negative means higher error.
        Percentages compare each model against its own target-only forecast on the same dataset.
        """),
        mo.Html('<div style="overflow-x: auto">' + covariate_effects.round(2).to_html(index=False, border=0, na_rep="—") + '</div>'),
    ])
    return


@app.cell
def _(mo, provenance):
    mo.accordion({"Result provenance": mo.Html(provenance.to_html(index=False, border=0))})
    return


@app.cell
def _(mo):
    mo.md("""
    One holdout per dataset illustrates behavior, not a general ranking. Public data may overlap model pretraining.
    Weather is ERA5 reanalysis for Strasbourg; retail is revenue rather than unit demand.
    Future observed weather is excluded; only calendar features are known into the future.
    Sources, covariates, and refresh commands: `timeseries/datasets/README.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
