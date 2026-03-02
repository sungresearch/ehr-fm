"""Tests for pretokenize features: interval injection, demographic prefix, and minor helpers.

Uses the same integration pattern as ``test_joint_mode_regression.py``: synthetic MEDS data
converted to MEDS Reader, then ``pretokenize_data`` with ``num_workers=0``.
"""

import json
from datetime import datetime, timedelta

import polars as pl
import pytest

from ehr_fm.defaults import (
    DEFAULT_DEMOGRAPHIC_LABELS,
    DEFAULT_INTERVAL_LABELS,
    REPEATABLE_INTERVAL_LABEL,
)
from ehr_fm.io import write_dict_to_json
from ehr_fm.tokenizer import _build_interval_token_lookup, pretokenize_data
from ehr_fm.vocabulary import prepend_demographic_tokens, prepend_interval_tokens

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _write_meds_and_convert(tmp_path, records, convert_meds_to_reader, *, n_patients=1):
    """Write MEDS parquet + metadata, return (reader_path, tmp_path)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    df = pl.DataFrame(records)
    for col, dtype in df.schema.items():
        if dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(pl.String))
    df.write_parquet(data_dir / "data_0.parquet")

    pids = sorted({r["subject_id"] for r in records})
    birth_time = datetime(2000, 1, 1)
    samples = pl.DataFrame(
        {
            "id": pids,
            "index_t": [birth_time + timedelta(days=3650)] * len(pids),
            "split": ["train"] * len(pids),
        }
    )
    samples.write_parquet(metadata_dir / "samples.parquet")

    codes = sorted({r["code"] for r in records})
    codes_df = pl.DataFrame({"code": codes, "description": codes})
    codes_df.write_parquet(metadata_dir / "codes.parquet")

    with open(metadata_dir / "dataset.json", "w") as f:
        json.dump({"dataset_name": "test", "dataset_version": "1.0"}, f)

    reader_path = convert_meds_to_reader(tmp_path)
    return reader_path


def _base_vocab_entries():
    """Minimal code-type vocab entries.

    Codes deliberately avoid "/" so that ``JointPolicy``'s ``remove_prefixes``
    (default=True) does not strip a prefix and cause a token-lookup mismatch.
    """
    return [
        {"type": "code", "code_string": "MEDS_BIRTH", "weight": -0.5},
        {"type": "code", "code_string": "EVENTCODE_A", "weight": -0.4},
        {"type": "code", "code_string": "EVENTCODE_B", "weight": -0.3},
        {"type": "code", "code_string": "8507", "weight": -0.2},
    ]


# ===================================================================
# _build_interval_token_lookup
# ===================================================================


class TestBuildIntervalTokenLookup:
    def test_basic(self):
        vocab = [
            {"type": "interval", "label": "INT_5m_15m"},
            {"type": "code", "label": "X"},
            {"type": "interval", "label": "INT_1h_2h"},
        ]
        lookup = _build_interval_token_lookup(vocab)
        assert lookup == {"INT_5m_15m": 0, "INT_1h_2h": 2}

    def test_no_intervals(self):
        vocab = [{"type": "code", "label": "X"}]
        assert _build_interval_token_lookup(vocab) == {}

    def test_missing_label(self):
        vocab = [{"type": "interval"}]
        assert _build_interval_token_lookup(vocab) == {}


# ===================================================================
# Interval injection integration
# ===================================================================


class TestIntervalInjection:
    @pytest.fixture
    def interval_dataset(self, tmp_path, convert_meds_to_reader):
        """MEDS Reader dataset with events at controlled time gaps + vocab with interval tokens."""
        birth_time = datetime(2000, 1, 1)

        records = [
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # 2 hours after birth -> should land in INT_1h_2h bin
            {
                "subject_id": 1,
                "time": birth_time + timedelta(hours=1, minutes=30),
                "code": "EVENTCODE_A",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # 1 year later -> triggers repeatable 6mt tokens + residual
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=365),
                "code": "EVENTCODE_B",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]

        reader_path = _write_meds_and_convert(tmp_path, records, convert_meds_to_reader)

        base = _base_vocab_entries()
        target_size = len(base) + len(DEFAULT_INTERVAL_LABELS)
        vocab_with_intervals = prepend_interval_tokens(base, target_vocab_size=target_size)
        vocab_data = {
            "vocab": vocab_with_intervals,
            "age_stats": {"mean": 2592000.0, "std": 1296000.0},
        }
        vocab_path = tmp_path / "vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)
        return reader_path, vocab_path, vocab_data

    def test_medium_gap_inserts_bin_token(self, interval_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = interval_dataset
        out_dir = tmp_path / "out_medium"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            inject_time_intervals=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.filter(pl.col("subject_id") == 1).row(0, named=True)
        token_ids = row["token_ids"]

        interval_type_ids = {i for i, v in enumerate(vocab_data["vocab"]) if v.get("type") == "interval"}
        has_interval = any(t in interval_type_ids for t in token_ids)
        assert has_interval, "Should contain at least one interval token for the 2-hour gap"

    def test_long_gap_repeatable_6mt(self, interval_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = interval_dataset
        out_dir = tmp_path / "out_long"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            inject_time_intervals=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.filter(pl.col("subject_id") == 1).row(0, named=True)
        token_ids = row["token_ids"]

        rep_label = REPEATABLE_INTERVAL_LABEL
        rep_id = next(i for i, v in enumerate(vocab_data["vocab"]) if v.get("label") == rep_label)
        rep_count = sum(1 for t in token_ids if t == rep_id)
        assert rep_count >= 1, "365-day gap should produce at least one repeatable 6mt token"

    def test_max_interval_repeat_caps(self, interval_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = interval_dataset
        out_dir = tmp_path / "out_capped"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            inject_time_intervals=True,
            max_interval_repeat=1,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.filter(pl.col("subject_id") == 1).row(0, named=True)
        token_ids = row["token_ids"]

        rep_label = REPEATABLE_INTERVAL_LABEL
        rep_id = next(i for i, v in enumerate(vocab_data["vocab"]) if v.get("label") == rep_label)
        rep_count = sum(1 for t in token_ids if t == rep_id)
        assert rep_count <= 1, "max_interval_repeat=1 should cap repeatable 6mt tokens"

    def test_token_length_alignment(self, interval_dataset, tmp_path):
        reader_path, vocab_path, _ = interval_dataset
        out_dir = tmp_path / "out_align"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            inject_time_intervals=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        for row in df.iter_rows(named=True):
            assert len(row["token_ids"]) == len(row["age"]) == len(row["age_normalized"]) == row["length"]


class TestIntervalInjectionShortGap:
    """Events < 5 minutes apart should NOT produce interval tokens."""

    @pytest.fixture
    def short_gap_dataset(self, tmp_path, convert_meds_to_reader):
        birth_time = datetime(2000, 1, 1)
        records = [
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(minutes=2),
                "code": "EVENTCODE_A",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(minutes=4),
                "code": "EVENTCODE_B",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]
        reader_path = _write_meds_and_convert(tmp_path, records, convert_meds_to_reader)
        base = _base_vocab_entries()
        target_size = len(base) + len(DEFAULT_INTERVAL_LABELS)
        vocab_entries = prepend_interval_tokens(base, target_vocab_size=target_size)
        vocab_data = {"vocab": vocab_entries, "age_stats": {"mean": 2592000.0, "std": 1296000.0}}
        vocab_path = tmp_path / "vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)
        return reader_path, vocab_path, vocab_data

    def test_no_interval_for_short_gap(self, short_gap_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = short_gap_dataset
        out_dir = tmp_path / "out_short"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            inject_time_intervals=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.row(0, named=True)
        interval_ids = {i for i, v in enumerate(vocab_data["vocab"]) if v.get("type") == "interval"}
        interval_count = sum(1 for t in row["token_ids"] if t in interval_ids)
        assert interval_count == 0, "Sub-5-min gaps should not produce interval tokens"


# ===================================================================
# Demographic prefix integration
# ===================================================================


class TestDemographicPrefix:
    @pytest.fixture
    def demo_dataset(self, tmp_path, convert_meds_to_reader):
        """MEDS Reader dataset with birth, sex code, and medical events + demographic vocab."""
        birth_time = datetime(2000, 1, 1)
        records = [
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # Sex code (male default = "8507")
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "8507",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            # Medical events at age ~0 (first event year = 2000)
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=30),
                "code": "EVENTCODE_A",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time + timedelta(days=60),
                "code": "EVENTCODE_B",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]
        reader_path = _write_meds_and_convert(tmp_path, records, convert_meds_to_reader)

        base = _base_vocab_entries()
        target_size = len(base) + len(DEFAULT_DEMOGRAPHIC_LABELS)
        vocab_entries = prepend_demographic_tokens(base, target_vocab_size=target_size)
        vocab_data = {"vocab": vocab_entries, "age_stats": {"mean": 2592000.0, "std": 1296000.0}}
        vocab_path = tmp_path / "vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)
        return reader_path, vocab_path, vocab_data

    def _lookup(self, vocab_data, label):
        for i, v in enumerate(vocab_data["vocab"]):
            if v.get("label") == label:
                return i
        return None

    def test_happy_path_sex_and_age(self, demo_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = demo_dataset
        out_dir = tmp_path / "out_demo"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            demographic_prefix=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.row(0, named=True)
        token_ids = row["token_ids"]

        sex_id = self._lookup(vocab_data, "DEMO_SEX_MALE")
        age_id = self._lookup(vocab_data, "DEMO_AGE_0_4")
        assert token_ids[0] == sex_id, "First token should be DEMO_SEX_MALE"
        assert token_ids[1] == age_id, "Second token should be DEMO_AGE_0_4"

    def test_with_year_token(self, demo_dataset, tmp_path):
        reader_path, vocab_path, vocab_data = demo_dataset
        out_dir = tmp_path / "out_demo_year"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            demographic_prefix=True,
            demographic_include_year=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.row(0, named=True)
        token_ids = row["token_ids"]

        year_id = self._lookup(vocab_data, "DEMO_YEAR_2000_2004")
        assert token_ids[2] == year_id, "Third token should be DEMO_YEAR_2000_2004"

    def test_age_positions_sequential(self, demo_dataset, tmp_path):
        reader_path, vocab_path, _ = demo_dataset
        out_dir = tmp_path / "out_demo_ages"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            demographic_prefix=True,
        )
        df = pl.read_parquet(out_dir / "patients_tokenized.parquet")
        row = df.row(0, named=True)
        ages = row["age"]
        ages_norm = row["age_normalized"]

        assert ages[0] == 0.0
        assert ages[1] == 1.0
        for a in ages_norm[:2]:
            assert a == 0.0

    def test_mutual_exclusion_with_max_events(self, demo_dataset, tmp_path):
        reader_path, vocab_path, _ = demo_dataset
        out_dir = tmp_path / "out_demo_clash"
        with pytest.raises(
            ValueError,
            match="max_events_per_patient.*demographic_prefix|demographic_prefix.*max_events_per_patient",
        ):
            pretokenize_data(
                vocab_path=vocab_path,
                out_dir=out_dir,
                dataset_path=reader_path,
                samples_path=reader_path / "metadata" / "samples.parquet",
                split="train",
                num_workers=0,
                demographic_prefix=True,
                max_events_per_patient=10,
            )


class TestDemographicPrefixSkips:
    """Samples that should be skipped when using demographic prefix."""

    @pytest.fixture
    def only_birth_dataset(self, tmp_path, convert_meds_to_reader):
        """Patient with only MEDS_BIRTH and sex code -- no medical events after."""
        birth_time = datetime(2000, 1, 1)
        records = [
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
            {
                "subject_id": 1,
                "time": birth_time,
                "code": "8507",
                "numeric_value": None,
                "text_value": None,
                "description": "",
                "unit": None,
            },
        ]
        reader_path = _write_meds_and_convert(tmp_path, records, convert_meds_to_reader)

        base = _base_vocab_entries()
        target_size = len(base) + len(DEFAULT_DEMOGRAPHIC_LABELS)
        vocab_entries = prepend_demographic_tokens(base, target_vocab_size=target_size)
        vocab_data = {"vocab": vocab_entries, "age_stats": {"mean": 2592000.0, "std": 1296000.0}}
        vocab_path = tmp_path / "vocab.json"
        write_dict_to_json(vocab_data, vocab_path, overwrite=True)
        return reader_path, vocab_path

    def test_no_medical_events_produces_empty_output(self, only_birth_dataset, tmp_path):
        reader_path, vocab_path = only_birth_dataset
        out_dir = tmp_path / "out_skip"
        pretokenize_data(
            vocab_path=vocab_path,
            out_dir=out_dir,
            dataset_path=reader_path,
            samples_path=reader_path / "metadata" / "samples.parquet",
            split="train",
            num_workers=0,
            demographic_prefix=True,
        )
        output_path = out_dir / "patients_tokenized.parquet"
        if output_path.exists():
            df = pl.read_parquet(output_path)
            assert len(df) == 0, "Patient with no medical events should be skipped"
        # If no parquet file was produced at all, that also counts as skipped
