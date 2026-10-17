import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline, mo, np, plt, torch


@app.cell
def _(mo):
    mo.md("""
    # Chronos-2: adding covariates
    **Covariates** are extra signals that may help a forecast. A sensor is available only in the past; a planned schedule is known in advance. These calls still use the pretrained model without fine-tuning.
    """)
    return


@app.cell
def _(np):
    time = np.arange(216)
    sensor = np.sin(time / 11)
    schedule = ((time % 48 >= 12) & (time % 48 < 24)).astype(float)
    signal = 0.03 * time + np.sin(2 * np.pi * time / 24) + 0.7 * sensor + 2 * schedule
    history = signal[:192]
    past_sensor = sensor[:192]  # The model never receives the future sensor readings.
    return history, past_sensor, schedule, signal, time


@app.cell
def _(Chronos2Pipeline, torch):
    model = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        device_map="cpu", dtype=torch.float32,
    )
    return (model,)


@app.cell
def _(history, model):
    # 1. Target history only.
    zero_quantiles, _ = model.predict_quantiles(
        inputs=[history], prediction_length=24, quantile_levels=[0.5],
    )
    zero_shot = zero_quantiles[0][0, :, 0].numpy()
    return (zero_shot,)


@app.cell
def _(history, model, past_sensor):
    # 2. Add a covariate available only over the history.
    past_quantiles, _ = model.predict_quantiles(
        inputs=[{"target": history, "past_covariates": {"sensor": past_sensor}}],
        prediction_length=24, quantile_levels=[0.5],
    )
    past_only = past_quantiles[0][0, :, 0].numpy()
    return (past_only,)


@app.cell
def _(history, model, past_sensor, schedule):
    # 3. A known-future covariate also needs its historical values.
    future_quantiles, _ = model.predict_quantiles(
        inputs=[{
            "target": history,
            "past_covariates": {"sensor": past_sensor, "schedule": schedule[:192]},
            "future_covariates": {"schedule": schedule[192:]},
        }],
        prediction_length=24, quantile_levels=[0.5],
    )
    with_future = future_quantiles[0][0, :, 0].numpy()
    return (with_future,)


@app.cell
def _(past_only, plt, signal, time, with_future, zero_shot):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(time[168:192], signal[168:192], color="#777777", label="History")
    ax.plot(time[192:], signal[192:], "o-", color="#111111", label="Actual", markersize=3)
    ax.plot(time[192:], zero_shot, ":", color="#777777", label="Target only")
    ax.plot(time[192:], past_only, "--", color="#2364AA", label="Past covariate")
    ax.plot(time[192:], with_future, color="#C15F14", label="Past + known future")
    ax.set(title="Chronos-2 · covariates", xlabel="Time step", ylabel="Synthetic value")
    ax.legend(fontsize=9)
    fig
    return (fig,)


@app.cell
def _(mo):
    mo.md("""
    This toy example explains the API, not model quality. Covariates do not guarantee better accuracy.
    """)
    return




if __name__ == "__main__":
    app.run()
