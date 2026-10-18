# Chronos-2

Start with `quick_start.py`, then `covariates.py`. Both show direct native API calls. `evaluate.py` is the thin adapter used by the shared runner.

Checkpoint: `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. The installed library is pinned by this folder's uv lockfile. See the [model card](https://huggingface.co/amazon/chronos-2) for training and architecture details.

Historical targets enter a pretrained forecasting pipeline; historical covariates and known-future calendar inputs can be added without parameter fitting here. The native quantile output is normalized to one median and p10/p50/p90 per future step. The quick start makes the shapes visible before the adapter handles real task windows.

Runs use CPU float32, one entity per call, exact task context, `cross_learning=False`, and native batch-size setting 256. The checkpoint's context limit is checked before inference; history is never silently shortened. Quantiles are native outputs, not calibrated coverage guarantees.

These results compare a checkpoint plus its library/inference policy. No fine-tuning occurs. Exposure to these public sources during pretraining is unknown. Timings exclude loading/downloads and are descriptive.
