# Regional electricity consumption

**Lesson:** consistency across a panel, daily/weekly baselines, uncertainty, and local calendars on a regular UTC grid.

Source: [RTE regional éCO2mix consolidated and definitive data](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/), Etalab Open Licence 2.0. The frozen extract spans 2023-11-13 through 2025-12-22 inclusive.

One observation is hourly mean power in MW: average the two half-hourly readings at :00/:30, requiring both. Do not sum MW readings and label them energy. Six regions are forecast separately: Grand Est, Bretagne, Île-de-France, Occitanie, Hauts-de-France, and Provence-Alpes-Côte d'Azur. Separate inference does not make their outcomes statistically independent.

Each forecast uses 1,344 hours (eight weeks) to predict 24. Predeclared origins are 00:00 UTC on the 8th/22nd monthly. Completeness exclusions leave **120 development windows in 2024** and **126 reserved evaluation windows in 2025**. Only development forecasts are permitted. Dates are never replaced based on model errors.

Both target history and forecast targets must be complete. The source clock audit discarded 24 nonexistent spring local labels only after finding valid counterparts at the same UTC keys. Remaining duplicate keys are rejected. The two autumn missing hours per region remain missing. Exact queries, response hashes, correction examples, revision categories, and excluded windows are retained in `rte_hourly.json`.

Baselines are persistence, daily naïve (24), weekly naïve (168, the fixed reference), and a four-week same-phase average. MASE uses mean absolute weekly differences in each permitted history; a zero scale is undefined. Origins are averaged within regions, then regions equally. Fixed UTC lags do not preserve local clock hour across daylight-saving changes.

Calendar inputs are numeric Europe/Paris hour and weekday, derived from UTC timestamps. No future weather, holidays, generation, or other regions' histories are supplied. Calendar-only changes the information track, not the target schedule or context.

Consolidated/definitive observations are retrospective revisions, not reconstructed real-time releases. The 2025 targets remain unscored and unplotted. Public-data pretraining exposure is unknown; development gains do not establish performance on the reserved period.
