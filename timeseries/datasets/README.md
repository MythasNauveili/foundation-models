# Reference datasets

Small public-data samples retrieved **2026-10-17**, committed so both notebooks use identical inputs. Refreshes may change historical values; the committed CSVs are the reference. Calendar covariates are computed in each notebook.

| CSV | Context | Holdout | Target | Seasonal lag |
|---|---|---|---|---|
| `strasbourg_daily.csv` | 730 days: 2023-12-26–2025-12-24 | 2025-12-25–2025-12-31 (7 days) | Daily maximum temperature, °C | 365 days |
| `bikes_hourly.csv` | 672 hours: 2012-12-03–2012-12-30 | 2012-12-31, 00:00–23:00 (24 hours) | Hourly rental count | 168 hours |
| `retail_monthly.csv` | 120 months: 2015-07–2025-06 | 2025-07–2025-12 (6 months) | Retail revenue, million USD | 12 months |

## Sources and covariates

- **Weather:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), explicitly selecting **ERA5**, latitude **48.58392**, longitude **7.74553**, and `Europe/Paris` calendar days. This is gridded reanalysis near Strasbourg, not a local station's measurements. Target: `temperature_2m_max`. Past-only covariates: `precipitation_sum` (mm/day), `wind_speed_10m_max` (km/h). Known over the forecast horizon: numeric day-of-year. Attribution: Open-Meteo and the Copernicus Climate Change Service/ECMWF ERA5 dataset; [CC BY 4.0](https://open-meteo.com/en/terms).
- **Bike demand:** [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T, H. (2013), [DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Extracted from `hour.csv`; target is `cnt`. Past-only `temp`, `hum`, and `windspeed` retain the source's normalized values. Known covariates: hour and weekday (Monday=0). Timestamps preserve source local clock labels without timezone conversion. The selected sample contains two missing historical hours, kept as empty values after hourly reindexing, not zeros. `casual` and `registered` are excluded because they compose the target. The holdout is the last complete calendar day of 2012.
- **Retail:** U.S. Census Bureau, [Retail Sales: Retail Trade, MRTSSM44000USN](https://fred.stlouisfed.org/series/MRTSSM44000USN), retrieved through FRED, Federal Reserve Bank of St. Louis. Monthly, **not seasonally adjusted**, nominal million USD; revenue is a demand proxy, not unit sales. Month-start timestamps label the entire calendar month's total. Known covariate: numeric month-of-year. U.S. Census source data are public domain; retain this attribution. Later releases may revise these observations.

## Comparison rules

Both models receive the same history, dates, horizon, and covariate values. The target-only run uses no covariates. The second run adds the listed past and calendar covariates; only calendar features extend into the forecast horizon. Holdout targets and observed future weather never enter inference.

Metrics use the median forecast. **MAE** is the mean absolute error; **bias** is the mean forecast minus actual, so positive bias means overforecasting. Both are in the target's units. For bike demand and retail, also report [Nicolas Vandeput's VN1 score](https://supchains.com/vn1-forecasting-competition/), using all timestamps in the holdout:

- `MAE (%) = 100 × sum(abs(forecast − actual)) / sum(actual)`
- `Bias (%) = 100 × sum(forecast − actual) / sum(actual)`
- `Score (%) = MAE (%) + abs(Bias (%))`; smaller is better.

MAE (%) is not MAPE: divide the total absolute error by total actual demand, rather than averaging individual percentage errors. Percentage metrics require nonnegative actuals and a positive total; otherwise they are left undefined. Omit them for weather because [Celsius has an arbitrary zero](https://otexts.com/fpp3/accuracy.html). Score the seasonal baseline with the same rules. Compare results within each holdout; do not average raw MAE across units or pool demand totals with different units.

Shaded bands are the model's 10th–90th percentiles. The seasonal naïve baseline repeats the corresponding historical values at the lag above. If a reference value is missing, it uses the most recent available earlier season at the same phase (one bike baseline point uses two weeks earlier). Inference timings exclude downloads and model loading, without warm-up or repeated timing trials. Model-specific inference defaults remain enabled.

One fixed holdout per dataset demonstrates the APIs; it cannot establish general model rankings. Public examples may overlap model pretraining data. ERA5 and revised retail observations are retrospective reference data, not a reconstruction of information available in real time.

## Refresh samples

From the repository root, using either model's environment:

```sh
uv run --project timeseries/chronos2 python timeseries/datasets/prepare.py
```

The script downloads from the official sources, trims fixed windows, preserves gaps, checks complete evaluation windows, and overwrites the three CSVs. Review changes and update the retrieval date above before committing a refresh. Normal notebook use makes no dataset requests. Refreshing data makes existing prediction files stale until they are regenerated.


## Refresh model results

Run each script in its own environment, from the repository root:

```sh
uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py
uv run --project timeseries/timestral python timeseries/timestral/evaluate.py
```

Each script loads its pinned checkpoint once and writes only its own files under `timeseries/results/`: a CSV containing 74 forecast rows (three datasets, two modes), and a JSON file recording the checkpoint revision, model-library version, generation time, CPU/dtype, dataset checksums, prediction-file checksum, and per-call durations. Keep these small reference results in Git; model weights and caches stay outside Git.

The shared `timeseries/evaluation.py` notebook calculates metrics from those files and displays both covariate modes in one section per dataset. A final recap lists the lowest MAEs, the demand scores, and the effect of covariates for every model/dataset pair, including the seasonal baseline where applicable. Raw errors are never averaged across datasets with different units. Dataset or prediction checksum mismatches stop comparisons and show the regeneration commands. Its lightweight uv project has no model libraries; use it to inspect results without downloading checkpoints. Model-specific inference defaults remain unchanged, including Timestral's symmetric averaging and automatic nonnegative clipping.
