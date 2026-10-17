import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from timestral import TimestralForecaster

    return TimestralForecaster, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Timestral-209M: your first forecast
    A **pretrained** model has already learned from other series; **zero-shot** means forecasting this new series without training on it.
    """)
    return


@app.cell
def _(np):
    time = np.arange(216)
    signal = 0.03 * time + np.sin(2 * np.pi * time / 24)
    history = signal[:192]  # Only these observations go into the model.
    return history, signal, time


@app.cell
def _(TimestralForecaster):
    model = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M",
        revision="9874d252ab0aae16f7f7c999f98f98d6ad7d8365", device="cpu",
    )
    return (model,)


@app.cell
def _(history, model):
    forecast = model.forecast(context=history, prediction_length=24)
    prediction = forecast.median[0, 0].numpy()
    return (prediction,)


@app.cell
def _(history, plt, prediction, signal, time):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(time[:192], history, color="#555555", label="History")
    ax.plot(time[192:], signal[192:], ":", color="#111111", label="Actual continuation")
    ax.plot(time[192:], prediction, "--", color="#2364AA", label="Forecast (median)")
    ax.set(title="Timestral-209M · 24 steps ahead", xlabel="Time step", ylabel="Synthetic value")
    ax.legend()
    fig
    return (fig,)




if __name__ == "__main__":
    app.run()
