# Two-hour bike demand

**Lesson:** daily/weekly structure, historical gaps, and disentangling calendar from observed weather.

One observation is aggregate rentals in a two-hour bin from the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T (2013), [DOI](https://doi.org/10.24432/C5W894), CC BY 4.0. Hourly counts are summed; normalized temperature, humidity, and wind are averaged. Source local clock labels are preserved without timezone conversion.

The frozen 2011–2012 extract contains 121 missing bins. Missing means unknown, not zero. Libraries use their native historical missing-value handling; there is no interpolation across the forecast boundary. Future preparation requires two observations per column independently so missing weather does not erase a measured target.

The task uses 4,380 two-hour observations (365 days) to predict 12 observations (one day). Preserve the 12 existing origins in 2012, selected as each month's last complete target day. February uses February 28. These previously inspected dates are **exploratory**. Eligibility uses held-out targets, not unused future weather.

Persistence, weekly naïve (lag 84), and a four-week same-phase average are shown. Weekly naïve is the fixed reference. Missing seasonal references fall back to earlier observations at the same phase; the four-week average uses only its four references.

Calendar inputs are numeric hour and weekday (Monday=0). Past-observed inputs are temperature, humidity, and wind speed. Never supply future realized weather or the `casual`/`registered` columns that compose the target.

This is one historical aggregate series, not station-level demand. Aggregation smooths peaks; selected complete days do not represent every calendar condition. Native missing-value policies are part of the compared pipelines. Pretraining exposure is unknown.
