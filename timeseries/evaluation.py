import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd
    from foundation_timeseries.tasks import AREA, DATASETS, TASKS, TRACKS, task_label
    from foundation_timeseries.results import load_comparison

    plt.rcParams.update({"font.size": 11})

    def table(frame):
        return mo.Html('<div style="overflow-x:auto">' +
                       frame.round(3).to_html(index=False, border=0, na_rep="—") + '</div>')

    return AREA, DATASETS, TASKS, TRACKS, load_comparison, mdates, mo, pd, plt, table, task_label


@app.cell
def _(TASKS, mo, task_label):
    task_picker = mo.ui.dropdown(
        options={task_label(t): key for key, t in TASKS.items()},
        value=task_label(TASKS[str(mo.cli_args().get("task", "weather-daily-7-v1"))]),
        allow_select_none=False, label="Forecasting task",
    )
    mo.vstack([mo.md("""
    # What does a pretrained model add?

    Compare a forecast with a simple baseline, then explore other inputs and uncertainty.
    **Context** is the observed history; **horizon** is how far ahead we predict.
    **Zero-shot** means no task-specific training here; pretraining exposure may be unknown.
    This notebook reads included results without downloading data or loading model weights.
    """), task_picker])
    return (task_picker,)


@app.cell
def _(AREA, DATASETS, TASKS, load_comparison, mo, task_picker):
    task = TASKS[task_picker.value]
    dataset = DATASETS[task.dataset]
    _issue = ""
    try:
        comparison = load_comparison(task)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as _error:
        _issue = str(_error)
    mo.stop(bool(_issue), mo.callout(f"Data unavailable: {_issue}. See the dataset guide.", kind="warn"))
    data, schedule = comparison.data, comparison.schedule
    _notices = [mo.md(f"""## {dataset.title}
    Context: **{task.history_length:,} observations** → horizon: **{task.horizon} steps**.
    One value is measured in **{dataset.unit}**. Scores cover **{len(schedule)} windows**
    in the **{task.default_split}** split; every history ends before its forecast starts.
    """)]
    if comparison.issues:
        _notices.append(mo.callout("Unavailable results: " + "; ".join(comparison.issues), kind="warn"))
    if comparison.predictions.empty:
        _notices.append(mo.callout("No valid model forecasts. Baselines remain available; see the dataset guide to generate forecasts."))
    _notices.append(mo.accordion({"Data source and limitations": mo.md((AREA / dataset.card).read_text())}))
    mo.vstack(_notices)
    return comparison, data, dataset, schedule, task


@app.cell
def _(TRACKS, comparison, mo, task):
    track_picker = mo.ui.dropdown(
        options={TRACKS[t]: t for t in task.tracks}, value=TRACKS["target_only"],
        label="Inputs", allow_select_none=False,
    )
    _available = {} if comparison.predictions.empty else {
        f"{name} ({model_id})": model_id
        for model_id, name in comparison.predictions[["Model ID", "Model"]].drop_duplicates().itertuples(index=False, name=None)
    }
    model_picker = mo.ui.multiselect(
        options=_available, value=[label for label, key in _available.items() if key in ("chronos2", "timestral-5m")],
        label="Models shown",
    )
    show_intervals = mo.ui.checkbox(label="Show native 80% forecast intervals")
    mo.accordion({"Compare other models or inputs": mo.vstack([
        track_picker, model_picker, show_intervals,
        mo.md("Calendar inputs are known in advance; observed weather is historical only. Each input track uses the same windows. Model selection filters the display; scores are never recomputed on a subset of dates."),
    ])})
    return model_picker, show_intervals, track_picker


@app.cell
def _(comparison, dataset, mo, model_picker, table, task, track_picker):
    summary = comparison.summary[
        (comparison.summary.Track == track_picker.value) &
        (comparison.summary["Model ID"].str.startswith("baseline:") | comparison.summary["Model ID"].isin(model_picker.value))
    ].sort_values([task.primary_metric, "Model ID"])
    _columns = ["Model", task.primary_metric, "Improvement (%)", "Windows"]
    _text = (f"MAE is average absolute error in {dataset.unit}; MASE divides it by seasonal differences in history. "
             "Lower is better. Average origins within each entity, then entities equally. "
             f"Improvement compares aggregate {task.primary_metric} with {task.reference}.")
    _blocks = [mo.md("### Does the model beat a simple baseline?\n" + _text), table(summary[_columns])]
    if summary[task.primary_metric].isna().any():
        _blocks.append(mo.callout("Undefined scores (—) propagate through averages; missing windows are never dropped.", kind="warn"))
    mo.vstack(_blocks)
    return (summary,)


@app.cell
def _(comparison, dataset, mo, model_picker, schedule):
    _labels = dict(dataset.entity_labels)
    _entities = sorted(schedule.entity.unique())
    entity_picker = mo.ui.dropdown(
        options={_labels.get(e, e): e for e in _entities}, value=_labels.get(_entities[0], _entities[0]),
        label="Entity", allow_select_none=False,
    )
    _models = comparison.summary[comparison.summary["Model ID"].isin(model_picker.value)]
    _options = {f"{name} ({key})": key for name, key in
                _models[["Model", "Model ID"]].drop_duplicates().itertuples(index=False, name=None)}
    diagnostic_model = mo.ui.dropdown(options=_options, value=next(iter(_options), None), label="Diagnostic model")
    return diagnostic_model, entity_picker


@app.cell
def _(diagnostic_model, entity_picker, mo, schedule):
    _origins = schedule.loc[schedule.entity == entity_picker.value, "forecast_start"]
    date_picker = mo.ui.dropdown(
        options={str(start): start for start in _origins}, value=str(_origins.iloc[0]),
        label="Forecast starts", allow_select_none=False,
    )
    case_picker = mo.ui.dropdown(
        options={"Selected date": "selected", "Largest model error": "failure"},
        value="Selected date", label="Worked example", allow_select_none=False,
    )
    mo.accordion({"Inspect another date or a difficult case": mo.vstack([
        entity_picker, date_picker, case_picker, diagnostic_model,
        mo.md("The difficult case is selected after scoring using the model's largest error. It is a diagnostic, not an independent test. These controls never change aggregate scores."),
    ])})
    return case_picker, date_picker


@app.cell
def _(case_picker, comparison, dataset, date_picker, diagnostic_model, entity_picker, mdates, mo, model_picker, plt, show_intervals, summary, table, task, track_picker):
    _key = (entity_picker.value, date_picker.value)
    _note = ""
    if case_picker.value == "failure":
        _cases = comparison.scores[(comparison.scores.Track == track_picker.value) &
                                   (comparison.scores["Model ID"] == diagnostic_model.value)]
        _cases = _cases.dropna(subset=[task.primary_metric]).sort_values(
            [task.primary_metric, "entity", "forecast_start"], ascending=[False, True, True])
        if not _cases.empty:
            _worst = _cases.iloc[0]
            _key = (_worst.entity, _worst.forecast_start)
            _note = f"Selected after scoring: largest model {task.primary_metric} = {_worst[task.primary_metric]:.3f}."
        else:
            _note = "Select a model to inspect its largest error."
    _history, _future, _baselines = comparison.windows[_key]
    _fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _recent = _history.tail(3 * task.horizon)
    _ax.plot(_recent.timestamp, _recent.target, color="#888888", label="History")
    _ax.plot(_future.timestamp, _future.target, "o-", color="#111111", markersize=3, label="Actual")
    for _name, _forecast in _baselines.items():
        _ax.plot(_future.timestamp, _forecast, ":", linewidth=1.5, label=_name)
    if not comparison.predictions.empty:
        _selected = comparison.predictions[
            (comparison.predictions.entity == _key[0]) & (comparison.predictions.forecast_start == _key[1]) &
            (comparison.predictions.track == track_picker.value) & comparison.predictions["Model ID"].isin(model_picker.value)]
        for (_model_id, _name), _group in _selected.groupby(["Model ID", "Model"], sort=False):
            _line, = _ax.plot(_group.timestamp, _group.point, linewidth=2, label=_name)
            if show_intervals.value and {"p10", "p90"}.issubset(_group) and _group[["p10", "p90"]].notna().all().all():
                _ax.fill_between(_group.timestamp, _group.p10, _group.p90, color=_line.get_color(), alpha=0.1)
    _ax.set(ylabel=dataset.unit, xlabel="UTC" if dataset.utc else "Source calendar time",
            title=f"{dict(dataset.entity_labels).get(_key[0], _key[0])} · {track_picker.value} · {_key[1]:%Y-%m-%d %H:%M}")
    _ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_ax.xaxis.get_major_locator()))
    _ax.legend(fontsize=8, ncol=2)
    _chart = mo.as_html(_fig)
    plt.close(_fig)
    _scores = comparison.scores[(comparison.scores.entity == _key[0]) &
        (comparison.scores.forecast_start == _key[1]) & (comparison.scores.Track == track_picker.value) &
        comparison.scores["Model ID"].isin(summary["Model ID"])]
    mo.vstack([mo.md(_note), _chart, mo.accordion({"Scores for this window": table(_scores)})])
    return


@app.cell
def _(comparison, data, dataset, mo, pd, schedule, summary, table, task, track_picker):
    # Stop before reserved outcomes; only profile history and targets of the selected split.
    _end = schedule.forecast_start.max() + task.horizon * pd.tseries.frequencies.to_offset(task.frequency)
    _coverage = data[data.timestamp < _end].groupby("entity").agg(
        Rows=("target", "size"), Missing=("target", lambda x: x.isna().sum()),
        First=("timestamp", "min"), Last=("timestamp", "max")).reset_index()
    _columns = [c for c in ("WIS", "Scaled WIS", "Coverage80 (%)", "Width80") if c in summary]
    _uncertainty = summary.dropna(subset=["WIS"]) if "WIS" in summary else summary.iloc[:0]
    _timing = comparison.timings
    if not _timing.empty:
        _timing = _timing[(_timing.Track == track_picker.value) & _timing["Model ID"].isin(summary["Model ID"])]
        _timing = _timing.groupby(["Model ID", "Model"]).Seconds.agg(["size", "mean", "sum"]).reset_index()
    _regional = comparison.entities
    _regional = _regional[(_regional.Track == track_picker.value) & _regional["Model ID"].isin(summary["Model ID"])]
    mo.accordion({
        "Data coverage and scores by entity": mo.vstack([
            mo.md("Missing observations remain distinct from zero. Reserved outcomes are excluded."),
            table(_coverage), table(_regional)]),
        "Uncertainty": mo.vstack([
            mo.md("An 80% marginal interval concerns each timestamp, not the whole path. Coverage near 80% is a diagnostic, not a guarantee. WIS rewards narrow intervals and penalizes misses; lower is better."),
            table(_uncertainty[["Model", *_columns]]) if not _uncertainty.empty else mo.md("No probabilistic forecasts selected.")]),
        "Inference cost and provenance": mo.vstack([
            mo.md("Seconds per entity/origin call: size = calls, mean = seconds/call, sum = total seconds. CPU times exclude loading/downloads; they are descriptive, without controlled warm-up or repeated trials."),
            table(_timing), table(comparison.provenance)]),
    })
    return


if __name__ == "__main__":
    app.run()
