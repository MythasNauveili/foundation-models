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
    # Timestral: adding covariates
    **Covariates** are extra signals that may help a forecast. A sensor is available only in the past; a planned schedule is known in advance. These calls still use the pretrained model without fine-tuning.
    Change `size` below to use the same calls with any Timestral checkpoint.
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
    # 1. Target history only.
    zero_forecast = model.forecast(context=history, prediction_length=24)
    zero_shot = zero_forecast.median[0, 0].numpy()
    return (zero_shot,)


@app.cell
def _(history, model, past_sensor):
    # 2. Covariate arrays use (series, covariates, time).
    past_forecast = model.forecast(
        context=history, prediction_length=24,
        past_covariates=past_sensor.reshape(1, 1, 192),
    )
    past_only = past_forecast.median[0, 0].numpy()
    return (past_only,)


@app.cell
def _(history, model, past_sensor, schedule):
    # 3. Known covariates cover all 192 historical + 24 future steps.
    future_forecast = model.forecast(
        context=history, prediction_length=24,
        past_covariates=past_sensor.reshape(1, 1, 192),
        known_covariates=schedule.reshape(1, 1, 216),
    )
    with_future = future_forecast.median[0, 0].numpy()
    return (with_future,)


@app.cell
def _(past_only, plt, signal, size, time, with_future, zero_shot):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(time[168:192], signal[168:192], color="#777777", label="History")
    ax.plot(time[192:], signal[192:], "o-", color="#111111", label="Actual", markersize=3)
    ax.plot(time[192:], zero_shot, ":", color="#777777", label="Target only")
    ax.plot(time[192:], past_only, "--", color="#2364AA", label="Past covariate")
    ax.plot(time[192:], with_future, color="#C15F14", label="Past + known future")
    ax.set(title=f"Timestral-{size} · covariates", xlabel="Time step", ylabel="Synthetic value")
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
