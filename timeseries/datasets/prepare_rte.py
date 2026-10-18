"""Freeze the small RTE task, without running models or scoring reserved dates.

Run from the repository root: uv run --project timeseries python timeseries/datasets/prepare_rte.py
"""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from foundation_timeseries.tasks import AREA, REGIONS, TASKS, forecast_window, normalize_cycles

TASK = TASKS["rte-hourly-24-v1"]

SOURCE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def"
START, END = "2023-11-13T00:00:00Z", "2025-12-23T00:00:00Z"


def hourly_consumption(raw, start=START, end=END):
    """Validate half-hourly keys and average paired power readings on a UTC grid."""
    raw = raw.copy()
    raw["code_insee_region"] = raw.code_insee_region.astype(str)
    raw["timestamp"] = pd.to_datetime(raw.date_heure, utc=True)
    assert raw.timestamp.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="left").all()
    assert raw.timestamp.dt.minute.isin([0, 30]).all() and raw.timestamp.dt.second.eq(0).all()
    assert set(raw.code_insee_region).issubset(REGIONS), "unexpected region"
    raw = raw.drop_duplicates()
    assert not raw.duplicated(["code_insee_region", "timestamp"]).any(), (
        "conflicting half-hourly duplicates"
    )
    raw["consommation"] = pd.to_numeric(raw.consommation, errors="raise")
    present = raw.consommation.dropna()
    assert np.isfinite(present).all() and (present >= 0).all(), "invalid consumption"
    frames = []
    for entity, group in raw.groupby("code_insee_region", sort=False):
        target = group.set_index("timestamp").consommation.sort_index()
        target = target.reindex(pd.date_range(start, end, freq="30min", inclusive="left"))
        hourly = target.resample("h").mean().where(target.resample("h").count().eq(2))
        frames.append(
            pd.DataFrame({"entity": entity, "timestamp": hourly.index, "target": hourly.to_numpy()})
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["entity", "timestamp"])
        .reset_index(drop=True)
    )


def rte_cycles(data):
    eligible, excluded = [], []
    for entity in REGIONS:
        for year, split in [(2024, "development"), (2025, "evaluation")]:
            for month in range(1, 13):
                for day in [8, 22]:
                    row = {
                        "task": TASK.id,
                        "entity": entity,
                        "split": split,
                        "forecast_start": f"{year}-{month:02d}-{day:02d}T00:00:00+00:00",
                    }
                    cycle = next(
                        normalize_cycles(pd.DataFrame([row]), TASK).itertuples(index=False)
                    )
                    try:
                        forecast_window(data, cycle, TASK)
                    except (AssertionError, ValueError, KeyError) as error:
                        excluded.append({**row, "reason": str(error)})
                    else:
                        eligible.append(row)
    return pd.DataFrame(eligible), excluded


def correct_spring_labels(raw):
    """Audit a source DST defect: nonexistent local labels duplicate valid UTC keys."""
    utc = pd.to_datetime(raw.date_heure, utc=True)
    local = utc.dt.tz_convert("Europe/Paris")
    assert (local.dt.strftime("%Y-%m-%d") == raw.date).all(), (
        "source dates disagree with UTC timestamps"
    )
    labels = pd.to_datetime(raw.date + " " + raw.heure)
    nonexistent = labels.dt.tz_localize("Europe/Paris", ambiguous=True, nonexistent="NaT").isna()
    mismatch = local.dt.strftime("%H:%M") != raw.heure
    assert mismatch.eq(nonexistent).all(), "unexpected source clock mismatch"
    valid_keys = pd.MultiIndex.from_arrays(
        [raw.loc[~nonexistent, "code_insee_region"], utc[~nonexistent]]
    )
    invalid_keys = pd.MultiIndex.from_arrays(
        [raw.loc[nonexistent, "code_insee_region"], utc[nonexistent]]
    )
    assert invalid_keys.isin(valid_keys).all(), "invalid spring label without a valid counterpart"
    notes = {
        "timestamp_check": "date_heure is the authoritative timezone-aware coordinate; source dates agree with Europe/Paris.",
        "discarded_nonexistent_local_clock_rows": int(nonexistent.sum()),
        "clock_correction": "Discard nonexistent spring 02:00/02:30 local labels only when a valid 03:00/03:30 counterpart has the same UTC key; all remaining duplicate keys are rejected. Autumn gaps remain missing.",
        "clock_label_examples": raw.loc[
            nonexistent, ["code_insee_region", "date_heure", "date", "heure"]
        ]
        .head(4)
        .to_dict("records"),
    }
    return raw.loc[~nonexistent].copy(), notes


def main():
    frames, queries = [], []
    for code in REGIONS:
        query = urlencode(
            {
                "select": "code_insee_region,libelle_region,nature,date_heure,consommation,date,heure",
                "where": f'code_insee_region="{code}" AND date_heure >= "{START}" AND date_heure < "{END}"',
                "order_by": "date_heure",
                "timezone": "UTC",
                "use_labels": "false",
                "delimiter": ",",
            }
        )
        url = f"{SOURCE}/exports/csv?{query}"
        with urlopen(url, timeout=90) as response:
            payload = response.read()
        frame = pd.read_csv(BytesIO(payload), dtype={"code_insee_region": str})
        assert len(frame), f"No source rows for {code}"
        queries.append({"url": url, "raw_sha256": sha256(payload).hexdigest(), "rows": len(frame)})
        frames.append(frame)
        print(f"{code} · {REGIONS[code]}: {len(frame):,} half-hourly rows", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    assert set(raw.code_insee_region) == set(REGIONS)
    assert raw.groupby("code_insee_region").libelle_region.nunique().eq(1).all(), (
        "ambiguous region labels"
    )
    assert raw.nature.notna().all(), "missing revision category"
    assert set(raw.nature) <= {"Données définitives", "Données consolidées"}, (
        "unknown revision category"
    )
    corrected, clock_notes = correct_spring_labels(raw)
    hourly = hourly_consumption(corrected)
    eligible, excluded = rte_cycles(hourly)
    assert not eligible.empty, "no eligible RTE windows"
    destination = AREA / "datasets"
    csv_path = destination / "rte_hourly.csv"
    hourly.to_csv(csv_path, index=False, float_format="%.1f")
    development = hourly[hourly.timestamp.dt.year.eq(2024)]
    profile = (
        development.groupby("entity").target.agg(["count", "min", "max", "mean"]).reset_index()
    )
    metadata = {
        "source": "https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/",
        "licence": "Etalab Open Licence 2.0",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "start_inclusive": START,
        "end_exclusive": END,
        "regions": REGIONS,
        "source_labels": raw.groupby("code_insee_region").libelle_region.first().to_dict(),
        "revision_categories": raw.nature.value_counts().to_dict(),
        "raw_rows": len(raw),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        **clock_notes,
        "aggregation": "Mean of the two timestamped half-hourly MW values at :00 and :30; both required.",
        "hourly_rows": len(hourly),
        "missing_hours_by_entity": hourly.groupby("entity")
        .target.apply(lambda x: int(x.isna().sum()))
        .to_dict(),
        "development_profile": profile.to_dict("records"),
        "cycles": eligible.to_dict("records"),
        "eligible_windows": eligible.groupby("split").size().to_dict(),
        "excluded_windows": excluded,
        "availability": "Retrospective consolidated/definitive history; historical as-of availability is not reconstructed.",
        "pretraining_exposure": "unknown",
        "reserved_scores_computed": False,
        "dataset_sha256": sha256(csv_path.read_bytes()).hexdigest(),
    }
    # Preserve the old teaching schedules byte-for-value; only replace RTE rows.
    path = destination / "cycles.csv"
    previous = pd.read_csv(path, dtype={"entity": str})
    previous = previous.loc[previous.task != TASK.id].copy()
    pd.concat([previous, eligible], ignore_index=True).to_csv(path, index=False)
    metadata["schedule_sha256"] = sha256(path.read_bytes()).hexdigest()
    (destination / "rte_hourly.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "missing_hours": metadata["missing_hours_by_entity"],
                "eligible_windows": metadata["eligible_windows"],
                "excluded": len(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
