"""Generate synthetic EHR data in MEDS and MEDS Reader formats for the EHR-FM tutorial.

Creates a dataset of ~200 patients with realistic EHR event patterns including:
  - Birth events (MEDS_BIRTH)
  - Sex demographics (Gender/F, Gender/M)
  - Lab results with numeric values (glucose, hemoglobin, creatinine, potassium, WBC)
  - Diagnosis codes (diabetes, hypertension, pneumonia, etc.)
  - Drug orders with text values describing route of administration

Output structure::

    {destination}/
        meds/                       Standard MEDS parquet format
            data/
                0.parquet           All patient events
            metadata/
                dataset.json        Dataset metadata
                samples.parquet     Subject IDs, index times, train/validation splits
        meds_reader/                MEDS Reader format (converted and verified)
            code/
            description/
            numeric_value/
            text_value/
            time/
            subject_id
            meds_reader.*
            metadata/
                samples.parquet

The generated dataset is committed to the repository so tutorial users do not
need to re-run this script.  Only re-run if you want to regenerate the data.

Usage::

    python tutorials/generate_synthetic_data.py
    python tutorials/generate_synthetic_data.py --destination /custom/path
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ehr_fm.meds_reader_utils import convert_to_meds_reader, verify_meds_reader

# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------

LAB_RANGES: dict[str, tuple[float, float]] = {
    "LAB/glucose": (60.0, 300.0),
    "LAB/hemoglobin": (7.0, 18.0),
    "LAB/creatinine": (0.4, 6.0),
    "LAB/potassium": (2.5, 7.0),
    "LAB/wbc": (2.0, 20.0),
}

DIAGNOSIS_CODES: list[str] = [
    "DIAGNOSIS/diabetes",
    "DIAGNOSIS/hypertension",
    "DIAGNOSIS/pneumonia",
    "DIAGNOSIS/heart_failure",
    "DIAGNOSIS/copd",
    "DIAGNOSIS/uti",
    "DIAGNOSIS/anemia",
    "DIAGNOSIS/sepsis",
]

DRUG_ROUTES: dict[str, str] = {
    "DRUG/metformin": "oral",
    "DRUG/lisinopril": "oral",
    "DRUG/insulin": "subcutaneous",
    "DRUG/vancomycin": "intravenous",
    "DRUG/amoxicillin": "oral",
}

# ---------------------------------------------------------------------------
# Per-patient generator
# ---------------------------------------------------------------------------


def _generate_patient(subject_id: int) -> list[dict]:
    """Return a list of MEDS event dicts for one synthetic patient."""
    birth = datetime(1990, 1, 1) + timedelta(days=random.randint(0, 7300))

    events: list[dict] = []

    events.append(
        {
            "subject_id": subject_id,
            "time": birth,
            "code": "MEDS_BIRTH",
            "numeric_value": None,
            "text_value": None,
            "description": None,
        }
    )

    sex = random.choice(["Gender/F", "Gender/M"])
    events.append(
        {
            "subject_id": subject_id,
            "time": birth,
            "code": sex,
            "numeric_value": None,
            "text_value": None,
            "description": None,
        }
    )

    current_time = birth + timedelta(days=random.randint(365, 1825))
    num_encounters = random.randint(2, 8)

    for _ in range(num_encounters):
        num_clinical = random.randint(1, 6)
        for _ in range(num_clinical):
            event_time = current_time + timedelta(minutes=random.randint(0, 4320))
            event_type = random.choices(["lab", "diagnosis", "drug"], weights=[0.5, 0.3, 0.2])[0]

            if event_type == "lab":
                lab_code = random.choice(list(LAB_RANGES.keys()))
                lo, hi = LAB_RANGES[lab_code]
                events.append(
                    {
                        "subject_id": subject_id,
                        "time": event_time,
                        "code": lab_code,
                        "numeric_value": round(random.uniform(lo, hi), 1),
                        "text_value": None,
                        "description": None,
                    }
                )
            elif event_type == "diagnosis":
                events.append(
                    {
                        "subject_id": subject_id,
                        "time": event_time,
                        "code": random.choice(DIAGNOSIS_CODES),
                        "numeric_value": None,
                        "text_value": None,
                        "description": None,
                    }
                )
            else:
                drug_code = random.choice(list(DRUG_ROUTES.keys()))
                events.append(
                    {
                        "subject_id": subject_id,
                        "time": event_time,
                        "code": drug_code,
                        "numeric_value": None,
                        "text_value": DRUG_ROUTES[drug_code],
                        "description": None,
                    }
                )

        current_time += timedelta(days=random.randint(30, 365))

    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic EHR data in MEDS + MEDS Reader format",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=str(Path(__file__).resolve().parent / "synthetic_data"),
        help="Output directory (default: tutorials/synthetic_data)",
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=200,
        help="Number of synthetic patients (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    dest = Path(args.destination)

    # -- 1. Generate events ------------------------------------------------
    all_events: list[dict] = []
    for sid in range(args.num_patients):
        all_events.extend(_generate_patient(sid))

    df = pl.DataFrame(all_events)
    df = df.sort(["subject_id", "time"])

    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))

    # -- 2. Write MEDS parquet ---------------------------------------------
    meds_dir = dest / "meds"
    data_dir = meds_dir / "data"
    metadata_dir = meds_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df.write_parquet(data_dir / "0.parquet")

    # -- 3. samples.parquet (80/20 train / validation) ---------------------
    last_times = df.group_by("subject_id").agg(pl.col("time").max().alias("index_t"))

    patient_ids = list(range(args.num_patients))
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)

    samples_rows: list[dict] = []
    for pid in patient_ids:
        row = last_times.filter(pl.col("subject_id") == pid).row(0, named=True)
        split = "train" if patient_ids.index(pid) < split_idx else "validation"
        samples_rows.append({"id": pid, "index_t": row["index_t"], "split": split})

    samples_df = pl.DataFrame(samples_rows).sort("id")
    samples_df.write_parquet(metadata_dir / "samples.parquet")

    # -- 4. dataset.json ---------------------------------------------------
    metadata = {
        "dataset_name": "ehr_fm_synthetic",
        "dataset_version": "1.0",
        "etl_name": "generate_synthetic_data",
        "etl_version": "1.0",
    }
    with open(metadata_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"MEDS dataset written to {meds_dir}")
    print(f"  patients : {args.num_patients}")
    print(f"  events   : {len(df)}")

    # -- 5. Convert to MEDS Reader -----------------------------------------
    meds_reader_dir = dest / "meds_reader"
    print(f"Converting to MEDS Reader at {meds_reader_dir} ...")
    convert_to_meds_reader(meds_dir, meds_reader_dir)

    # -- 6. Verify ---------------------------------------------------------
    print("Verifying MEDS Reader conversion ...")
    verify_meds_reader(meds_dir, meds_reader_dir)
    print("Done. MEDS Reader verification passed.")


if __name__ == "__main__":
    main()
