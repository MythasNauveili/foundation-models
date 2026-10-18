# Timestral

Start with `quick_start.py`, then `covariates.py`; change `size` for a checkpoint study. `evaluate.py` keeps the native forecast call visible while the shared runner handles the experiment.

| Size | Immutable checkpoint revision |
| --- | --- |
| 5M | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| 28M | `180698bbf5a50dda3516aa6419798437fb03a430` |
| 123M | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| 209M | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

Repositories are `mistralai/Timestral-<size>`; the library Git revision remains pinned in this folder's uv project. See the [5M model card](https://huggingface.co/mistralai/Timestral-5M) and the [library](https://github.com/mistralai/timestral).

The pretrained model consumes patches of historical observations and produces quantiles. Past covariates cover history; known covariates cover history plus horizon. The native tensors are normalized to one median and p10/p50/p90 per future step. The library's native missing-value handling remains part of the pipeline.

CPU float32 runs preserve symmetric averaging, clipping to nonnegative forecasts when context qualifies, and the native real-value cap. The context limit and effective per-shot decoding setting are recorded in `runtime` from the loaded checkpoint. Longer native block decoding is not a new independently trained model. No context is silently truncated.

The routine command defaults to 5M; `--size all` explicitly requests the four-size study. Size is not a measure of usefulness: compare matched accuracy and observed inference time. No fine-tuning occurs, interval coverage is not guaranteed, and public-source pretraining exposure is unknown.
