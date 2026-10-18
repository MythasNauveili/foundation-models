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
    # Timestral: your first forecast
    A **pretrained** model has already learned from other series; **zero-shot** means forecasting this new series without training on it.
    **Context** is the 192 observed values; the **horizon** is the next 24 steps. The plotted output is one median per future step.
    All four sizes share this API. Start with 5M; change `size` below to try another checkpoint.
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
    size = "5M"  # Choose 5M, 28M, 123M, or 209M.
    checkpoints = {
        "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
        "28M": "180698bbf5a50dda3516aa6419798437fb03a430",
        "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
        "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
    }
    model = TimestralForecaster.from_pretrained(
        f"mistralai/Timestral-{size}", revision=checkpoints[size], device="cpu",
    )
    return model, size


@app.cell
def _(history, model):
    forecast = model.forecast(context=history, prediction_length=24)
    prediction = forecast.median[0, 0].numpy()
    return (prediction,)


@app.cell
def _(history, plt, prediction, signal, size, time):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(time[:192], history, color="#555555", label="History")
    ax.plot(time[192:], signal[192:], ":", color="#111111", label="Actual continuation")
    ax.plot(time[192:], [history[-1]] * 24, ":", color="#C15F14", label="Persistence baseline")
    ax.plot(time[192:], prediction, "--", color="#2364AA", label="Forecast (median)")
    ax.set(title=f"Timestral-{size} · 24 steps ahead", xlabel="Time step", ylabel="Synthetic value")
    ax.legend()
    fig
    return (fig,)




if __name__ == "__main__":
    app.run()
