# Reference datasets and forecast cycles

Public-data samples retrieved **2026-10-17** are included so every model uses identical inputs. Upstream data can be revised; these CSVs remain the reference.

| Sample | Included period | History per forecast | Horizon | Evaluation cycles | Seasonal baseline |
|---|---|---|---|---|---|
| `strasbourg_daily.csv` | 2023–2025, 1,096 daily rows | 730 days | 7 days | Last 7 days of every month in 2025: 12 cycles | Lag 365 days |
| `bikes_2hourly.csv` | 2011–2012, 8,772 two-hour rows | 365 days / 4,380 steps | 24 hours / 12 steps | Last complete day of each month in 2012: 12 cycles | Lag 84 steps / 1 week |
| `retail_monthly.csv` | 2014–2025, 144 monthly rows | 120 months | 6 months | January and July starts in 2024 and 2025: 4 cycles | Lag 12 months |

`cycles.csv` records each forecast start, history length, horizon, seasonal lag, and frequency. Both model scripts and the evaluation notebook read this schedule. February's bike cycle starts on **2012-02-28**; the other bike cycles use the month's final day. Retail cycles cover four separate half-years without overlap.

## Sources and preparation

- **Weather:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), explicitly selecting **ERA5**, latitude **48.58392**, longitude **7.74553**, and `Europe/Paris` calendar days. This is gridded reanalysis near Strasbourg, not a local station's measurements. Target: `temperature_2m_max`, °C. Past-only covariates: `precipitation_sum` (mm/day) and `wind_speed_10m_max` (km/h). Known covariate: numeric day-of-year. Attribution: Open-Meteo and Copernicus Climate Change Service/ECMWF ERA5; [CC BY 4.0](https://open-meteo.com/en/terms).
- **Bikes:** [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T, H. (2013), [DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Start with `hour.csv`, reindex missing hours, then use left-labeled two-hour bins: target is the sum of both `cnt` values; `temp`, `hum`, and `windspeed` are their means in the source's normalized units. If either hourly row is incomplete, the entire bin stays missing. There are **121 missing bins** across the two-year sample. Each timestamp labels `[timestamp, timestamp + 2 hours)` using source local clock labels without timezone conversion. Two-hour frequency fits a full year within [Chronos-2's 8,192-observation limit](https://huggingface.co/amazon/chronos-2). Known covariates are the starting hour and weekday (Monday=0). Exclude `casual` and `registered`, which compose the target. Evaluation days have all 24 original hours present. The notebook's overview shows daily totals only when all 12 bins are complete; inference always uses two-hour values.
- **Retail:** U.S. Census Bureau, [Retail Sales: Retail Trade, MRTSSM44000USN](https://fred.stlouisfed.org/series/MRTSSM44000USN), retrieved through FRED, Federal Reserve Bank of St. Louis. Target: monthly nominal million USD, **not seasonally adjusted**; revenue is a demand proxy rather than unit sales. Month-start timestamps label the entire month's total. Known covariate: numeric month-of-year. U.S. Census source data are public domain; retain this attribution. Later releases can revise observations.

## Comparison and ranking rules

At each cycle, every model receives the same fixed-length trailing history and forecast horizon. Later cycles use actual observations available before their start, including observations from earlier evaluation windows. No fitting occurs and previous predictions never become history. Calendar covariates are calculated from timestamps; only these known features extend into the future. Held-out targets and observed future weather never enter inference. Missing historical bike bins remain explicit; each model retains its native missing-value behavior.

The seasonal naïve baseline uses only history at the listed lag. If a reference is missing, use the most recent earlier season at the same phase. Stop clearly if no valid reference exists. Model-specific inference defaults remain enabled, including Timestral's symmetric averaging and automatic nonnegative clipping. Requested histories are checked against each checkpoint's context limit to prevent truncation.

Metrics use the median forecast and are calculated **within each cycle first**:

- **MAE:** mean absolute error, in the target's units.
- **Bias:** mean forecast minus actual; positive means overforecasting.
- **Demand MAE (%):** `100 × sum(abs(forecast − actual)) / sum(actual)`.
- **Demand Bias (%):** `100 × sum(forecast − actual) / sum(actual)`.
- **[Vandeput score (%)](https://supchains.com/vn1-forecasting-competition/):** `MAE (%) + abs(Bias (%))`.

The ranked tables report arithmetic means of per-cycle metrics, with every cycle weighted equally. Weather ranks by mean cycle MAE in °C; bikes and retail rank by mean cycle score. Apply the absolute bias penalty **before** averaging, so opposite biases in different cycles do not cancel that penalty. The baseline appears once alongside every model/mode combination. Sort unrounded values ascending; exact ties share a rank.

MAE (%) divides total absolute error by total demand; it is not MAPE. Percentage metrics require nonnegative actuals and positive total demand. Undefined cycle metrics remain undefined in the aggregate and sort last; never silently average fewer cycles. Weather has **no score or percentage metrics**, because [Celsius has an arbitrary zero](https://otexts.com/fpp3/accuracy.html). There are no baseline-relative scores, cross-dataset averages, or overall model rankings.

Bands are native 10th–90th percentile forecasts, not guaranteed coverage. Inference timings record each CPU call separately, excluding downloads and model loading, without warm-up or repeated trials. The notebook summarizes call count, mean duration, and total duration. Seasonal cycles demonstrate behavior rather than general model quality. Public data can overlap pretraining; ERA5 and revised retail data are retrospective sources rather than real-time data vintages.

## Refresh samples and results

From the repository root:

```sh
uv run --project timeseries/chronos2 python timeseries/datasets/prepare.py
uv run --project timeseries/chronos2 python timeseries/chronos2/evaluate.py
uv run --project timeseries/timestral python timeseries/timestral/evaluate.py
```

Preparation downloads the official sources, validates the windows, and overwrites the three samples and schedule. Review changes and update the retrieval date before committing a refresh. Changed data or schedules invalidate the saved predictions. Normal notebook use makes no data requests.

The Timestral command evaluates all four sizes sequentially in one environment; append `--size 28M` to refresh only that size (also accepts `5M`, `123M`, and `209M`). Each checkpoint loads once for **56 inference calls**, producing **504 forecast rows**. Across all five models, that is **280 calls** and **2,520 rows**. Keep the small result CSVs and JSON metadata in Git; environments, model weights, and caches stay outside Git.

Each model's result CSV contains `dataset`, `mode`, `forecast_start`, `timestamp`, one-based `step`, `actual`, `baseline`, `p10`, `p50`, and `p90`. Version **2** JSON metadata records the checkpoint revision, model-library version, generation time, CPU/dtype, dataset checksums, cycle schedule and checksum, prediction checksum, and durations by dataset/mode/forecast start. Files remain `chronos2.csv`/`.json` and `timestral-{5m,28m,123m,209m}.csv`/`.json` under `timeseries/results/`.

The presentation environment has no model libraries. The evaluation notebook checks completeness, timestamps, actuals, historical baselines, quantiles, schema version, and checksums. Missing, stale, or old single-holdout results stop comparisons and show refresh commands. Each section includes the period overview, all-cycle rankings, performance by cycle, and a selector for detailed forecasts. Selecting a cycle never changes aggregate rankings or triggers inference. The final recap lists winners separately by dataset.
