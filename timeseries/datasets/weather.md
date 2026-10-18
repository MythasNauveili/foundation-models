# Strasbourg daily temperature

**Lesson:** a familiar noisy target, annual seasonality, and the limits of historical covariates.

One observation is daily maximum temperature in °C near Strasbourg (48.58392, 7.74553), from ERA5 reanalysis through the [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api). Retain Open-Meteo/ERA5 attribution and applicable CC BY terms. Source extract: 2023–2025; calendar dates use Europe/Paris.

The task uses 730 historical days to predict seven days, at the existing 12 origins in 2025. These dates have already been inspected and are **exploratory**, not an untouched holdout. The frozen extract has no missing targets.

Persistence repeats the last temperature; the fixed reference repeats the historical annual pattern at lag 365. This fixed lag differs from the same calendar date around leap years.

Calendar inputs are numeric day-of-year. Past-observed inputs are historical precipitation and maximum wind speed; their future realized values are never supplied. Calendar-only and past-only tracks isolate these changes. Numeric day-of-year does not encode circular proximity.

Reanalysis reconstructs historical conditions; it is not an archived operational weather forecast. Beating a seasonal baseline does not establish operational weather skill. No historical as-of availability is reconstructed. Public-data pretraining exposure is unknown. Zero-shot here means no task-specific parameter fitting.
