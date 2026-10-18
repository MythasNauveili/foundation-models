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
    # Chronos-2: your first forecast
    A **pretrained** model has already learned from other series; **zero-shot** means forecasting this new series without training on it.
    **Context** is the 192 observed values; the **horizon** is the next 24 steps. The plotted output is one median per future step.
    """)
    return


@app.cell
def _(np):
    time = np.arange(216)
    signal = 0.03 * time + np.sin(2 * np.pi * time / 24)
    history = signal[:192]  # Only these observations go into the model.
    return history, signal, time


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
    quantiles, _ = model.predict_quantiles(
        inputs=[history], prediction_length=24, quantile_levels=[0.5],
    )
    prediction = quantiles[0][0, :, 0].numpy()
    return (prediction,)


@app.cell
def _(history, plt, prediction, signal, time):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(time[:192], history, color="#555555", label="History")
    ax.plot(time[192:], signal[192:], ":", color="#111111", label="Actual continuation")
    ax.plot(time[192:], [history[-1]] * 24, ":", color="#C15F14", label="Persistence baseline")
    ax.plot(time[192:], prediction, "--", color="#2364AA", label="Forecast (median)")
    ax.set(title="Chronos-2 · 24 steps ahead", xlabel="Time step", ylabel="Synthetic value")
    ax.legend()
    fig
    return


if __name__ == "__main__":
    app.run()
