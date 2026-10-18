"""Export pinned Chronos-2 forecasts through the shared task runner."""

from foundation_timeseries.runner import (
    ModelSpec,
    Prediction,
    argument_parser,
    model_provenance,
    run,
    selected_tasks,
)

SPEC = ModelSpec(
    "chronos2",
    "Chronos-2",
    "amazon/chronos-2",
    "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "chronos-forecasting",
    "chronos2",
    settings=(
        ("device", "cpu"),
        ("dtype", "float32"),
        ("cross_learning", False),
        ("context", "exact task history"),
        ("batch_size", 256),
    ),
)


class ChronosAdapter:
    def __init__(self):
        import torch
        from chronos import Chronos2Pipeline

        self.model = Chronos2Pipeline.from_pretrained(
            SPEC.checkpoint, revision=SPEC.revision, device_map="cpu", dtype=torch.float32
        )
        self.spec = SPEC
        self.runtime = {"max_context": self.model.model_context_length}
        self.hardware = {
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "entities_per_call": 1,
        }

    def predict(self, inputs):
        payload = {"target": inputs.target}
        past = {**inputs.past_covariates, **inputs.known_history}
        if past:
            payload["past_covariates"] = past
        if inputs.known_future:
            payload["future_covariates"] = inputs.known_future
        output, _ = self.model.predict_quantiles(
            inputs=[payload],
            prediction_length=inputs.horizon,
            quantile_levels=list(SPEC.quantiles),
            context_length=len(inputs.target),
            cross_learning=False,
            batch_size=256,
        )
        quantiles = output[0][0].numpy().astype(float)
        return Prediction(
            quantiles[:, 1], "median", {q: quantiles[:, i] for i, q in enumerate(SPEC.quantiles)}
        )


if __name__ == "__main__":
    args = argument_parser(__doc__).parse_args()
    run(
        selected_tasks(args),
        SPEC,
        ChronosAdapter,
        smoke=args.smoke,
        refresh=args.refresh,
        provenance=model_provenance(SPEC),
    )
