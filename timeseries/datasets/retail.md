# Monthly retail revenue

**Lesson:** long context, annual seasonality, and why a target's units and publication timing matter.

One observation is nominal U.S. monthly retail revenue in million USD, not seasonally adjusted: U.S. Census Bureau via [FRED MRTSSM44000USN](https://fred.stlouisfed.org/series/MRTSSM44000USN). Census data are public domain; retain attribution. The frozen extract covers 2014–2025 with no missing targets. Month-start timestamps label the full month.

The task uses 120 months to predict six, at January/July origins in 2024 and 2025. All four origins have previously been inspected and are **exploratory**. Persistence and annual seasonal naïve (the fixed reference, lag 12) provide the alternatives.

The calendar track supplies numeric month over both history and future. There are no past-observed external covariates, so empty past-only tracks are not generated. Numeric month does not represent circular proximity directly.

Revenue combines prices, quantities, and economic composition; it is not units sold. Revised historical data and publication delays limit operational interpretation: the preceding month's total might not have been available at the nominal origin. No release-vintage reconstruction is claimed. MAE is the primary score. Pretraining exposure is unknown.
