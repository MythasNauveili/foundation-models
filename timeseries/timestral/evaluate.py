"""Export pinned Timestral forecasts; 5M by default, all sizes explicitly."""

from functools import partial

import numpy as np

from foundation_timeseries.runner import (
    ModelSpec,
    Prediction,
    argument_parser,
    model_provenance,
    run,
    selected_tasks,
)

CHECKPOINTS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "180698bbf5a50dda3516aa6419798437fb03a430",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


def model_spec(size):
    return ModelSpec(
        f"timestral-{size.lower()}",
        f"Timestral-{size}",
        f"mistralai/Timestral-{size}",
        CHECKPOINTS[size],
        "timestral",
        "timestral",
        settings=(
            ("device", "cpu"),
            ("dtype", "float32"),
            ("symmetric_averaging", True),
            ("clip_nonnegative", True),
            ("quantile_real_cap_k", 10000.0),
            ("per_shot_horizon_patches", "checkpoint default"),
        ),
    )


class TimestralAdapter:
    def __init__(self, spec):
        import torch
        from timestral import TimestralForecaster

        self.model = TimestralForecaster.from_pretrained(
            spec.checkpoint, revision=spec.revision, device="cpu"
        )
        self.spec = spec
        self.runtime = {
            "max_context": self.model.max_context,
            "per_shot_horizon_patches": self.model.per_shot_horizon_patches,
        }
        self.hardware = {
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "entities_per_call": 1,
        }

    def predict(self, inputs):
        past = (
            np.stack(list(inputs.past_covariates.values()))[None, ...]
            if inputs.past_covariates
            else None
        )
        known = (
            np.stack(
                [
                    np.concatenate([inputs.known_history[c], inputs.known_future[c]])
                    for c in inputs.known_history
                ]
            )[None, ...]
            if inputs.known_history
            else None
        )
        output = self.model.forecast(
            context=inputs.target,
            prediction_length=inputs.horizon,
            past_covariates=past,
            known_covariates=known,
            symmetric_averaging=True,
            clip_nonnegative=True,
        )
        quantiles = {q: output.quantile(q)[0, 0].numpy().astype(float) for q in self.spec.quantiles}
        return Prediction(quantiles[0.5], "median", quantiles)


if __name__ == "__main__":
    args = argument_parser(__doc__, CHECKPOINTS).parse_args()
    for size in CHECKPOINTS if args.size == "all" else [args.size]:
        spec = model_spec(size)
        run(
            selected_tasks(args),
            spec,
            partial(TimestralAdapter, spec),
            smoke=args.smoke,
            refresh=args.refresh,
            provenance=model_provenance(spec),
        )
