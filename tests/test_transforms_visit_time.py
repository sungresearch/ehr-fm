from datetime import datetime

import polars as pl
import pytest

from ehr_fm.transforms.validation import VisitEndTimeConfig
from ehr_fm.transforms.visit_time import VisitEndTimeMover


def _build_visit_df(
    *,
    include_workflow_stage: bool = True,
    include_visit_id: bool = True,
    include_code: bool = True,
    include_description: bool = False,
) -> pl.DataFrame:
    """Build a synthetic MEDS DataFrame with visit start/end events."""
    visit_start = datetime(2024, 1, 1, 8, 0)
    visit_end = datetime(2024, 1, 3, 16, 0)

    data: dict[str, list] = {
        "subject_id": [1, 1, 1, 1, 1],
        "time": [visit_start, visit_end, visit_start, visit_start, visit_start],
    }
    if include_code:
        data["code"] = [
            "VISIT/IP",
            "VISIT/IP",
            "DIAGNOSIS/flu",
            "PROCEDURE/xray",
            "LAB/glucose",
        ]
    if include_visit_id:
        data["visit_id"] = [100, 100, 100, 100, 100]
    if include_workflow_stage:
        data["workflow_stage"] = ["start", "end", None, None, None]
    if include_description:
        data["description"] = [
            "Inpatient visit",
            "Inpatient visit",
            "DIAGNOSIS/flu desc",
            "Some procedure",
            "Lab test",
        ]
    return pl.DataFrame(data)


class TestVisitEndTimeMoverGuards:
    """Guard branches: missing columns return data unchanged."""

    def test_missing_workflow_stage(self):
        df = _build_visit_df(include_workflow_stage=False)
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        assert result.shape == df.shape
        assert (
            result["time"].to_list() == df.sort(["subject_id", "time"], maintain_order=True)["time"].to_list()
        )

    def test_missing_visit_id(self):
        df = _build_visit_df(include_visit_id=False)
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        assert result.shape == df.shape

    def test_missing_code(self):
        df = _build_visit_df(include_code=False)
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        assert result.shape == df.shape


class TestVisitEndTimeMoverNoVisits:
    def test_no_visit_events(self):
        df = pl.DataFrame(
            {
                "subject_id": [1, 1],
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "code": ["LAB/glucose", "DIAGNOSIS/flu"],
                "visit_id": [1, 1],
                "workflow_stage": pl.Series([None, None], dtype=pl.String),
            }
        )
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        assert result.shape == df.shape


class TestVisitEndTimeMoverHappyPath:
    @pytest.fixture
    def mover(self):
        return VisitEndTimeMover(VisitEndTimeConfig())

    @pytest.fixture
    def visit_df(self):
        return _build_visit_df()

    def test_diagnosis_moved_to_visit_end(self, mover, visit_df):
        result = mover(visit_df)
        visit_end = datetime(2024, 1, 3, 16, 0)
        diag_row = result.filter(pl.col("code") == "DIAGNOSIS/flu")
        assert diag_row["time"][0] == visit_end

    def test_procedure_moved_to_visit_end(self, mover, visit_df):
        result = mover(visit_df)
        visit_end = datetime(2024, 1, 3, 16, 0)
        proc_row = result.filter(pl.col("code") == "PROCEDURE/xray")
        assert proc_row["time"][0] == visit_end

    def test_lab_not_moved(self, mover, visit_df):
        """LAB/ is not in the default prefix list, so it should stay at visit start."""
        visit_start = datetime(2024, 1, 1, 8, 0)
        result = mover(visit_df)
        lab_row = result.filter(pl.col("code") == "LAB/glucose")
        assert lab_row["time"][0] == visit_start

    def test_row_count_unchanged(self, mover, visit_df):
        result = mover(visit_df)
        assert len(result) == len(visit_df)

    def test_result_sorted_by_subject_and_time(self, mover, visit_df):
        result = mover(visit_df)
        times = result["time"].to_list()
        subject_ids = result["subject_id"].to_list()
        pairs = list(zip(subject_ids, times))
        assert pairs == sorted(pairs)


class TestVisitEndTimeMoverDescriptionMatching:
    def test_description_prefix_moves_event(self):
        """When check_description=True, events matched by description are also moved."""
        visit_start = datetime(2024, 1, 1, 8, 0)
        visit_end = datetime(2024, 1, 3, 16, 0)

        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [visit_start, visit_end, visit_start],
                "code": ["VISIT/IP", "VISIT/IP", "SOME_CODE"],
                "visit_id": [100, 100, 100],
                "workflow_stage": ["start", "end", None],
                "description": ["visit", "visit", "DIAGNOSIS/from_description"],
            }
        )
        cfg = VisitEndTimeConfig(check_description=True)
        mover = VisitEndTimeMover(cfg)
        result = mover(df)
        moved_row = result.filter(pl.col("code") == "SOME_CODE")
        assert moved_row["time"][0] == visit_end

    def test_description_disabled_no_move(self):
        visit_start = datetime(2024, 1, 1, 8, 0)
        visit_end = datetime(2024, 1, 3, 16, 0)

        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [visit_start, visit_end, visit_start],
                "code": ["VISIT/IP", "VISIT/IP", "SOME_CODE"],
                "visit_id": [100, 100, 100],
                "workflow_stage": ["start", "end", None],
                "description": ["visit", "visit", "DIAGNOSIS/from_description"],
            }
        )
        cfg = VisitEndTimeConfig(check_description=False)
        mover = VisitEndTimeMover(cfg)
        result = mover(df)
        moved_row = result.filter(pl.col("code") == "SOME_CODE")
        assert moved_row["time"][0] == visit_start


class TestVisitEndTimeMoverMultiVisit:
    def test_two_visits_correct_end_times(self):
        v1_start = datetime(2024, 1, 1, 8, 0)
        v1_end = datetime(2024, 1, 2, 16, 0)
        v2_start = datetime(2024, 2, 1, 8, 0)
        v2_end = datetime(2024, 2, 3, 10, 0)

        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1],
                "time": [v1_start, v1_end, v1_start, v2_start, v2_end, v2_start],
                "code": [
                    "VISIT/IP",
                    "VISIT/IP",
                    "DIAGNOSIS/cold",
                    "VISIT/IP",
                    "VISIT/IP",
                    "DIAGNOSIS/flu",
                ],
                "visit_id": [10, 10, 10, 20, 20, 20],
                "workflow_stage": ["start", "end", None, "start", "end", None],
            }
        )
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)

        cold_row = result.filter(pl.col("code") == "DIAGNOSIS/cold")
        assert cold_row["time"][0] == v1_end

        flu_row = result.filter(pl.col("code") == "DIAGNOSIS/flu")
        assert flu_row["time"][0] == v2_end


class TestVisitEndTimeMoverEdgeCases:
    def test_event_not_at_visit_start_not_moved(self):
        """Events whose time differs from visit_start_time should NOT be moved."""
        visit_start = datetime(2024, 1, 1, 8, 0)
        visit_end = datetime(2024, 1, 3, 16, 0)
        different_time = datetime(2024, 1, 2, 12, 0)

        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "time": [visit_start, visit_end, different_time],
                "code": ["VISIT/IP", "VISIT/IP", "DIAGNOSIS/flu"],
                "visit_id": [100, 100, 100],
                "workflow_stage": ["start", "end", None],
            }
        )
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        diag_row = result.filter(pl.col("code") == "DIAGNOSIS/flu")
        assert diag_row["time"][0] == different_time

    def test_diagnosis_actually_moved(self):
        """Verify the DIAGNOSIS event's time changes but LAB stays the same."""
        visit_start = datetime(2024, 1, 1, 8, 0)
        visit_end = datetime(2024, 1, 3, 16, 0)

        df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "time": [visit_start, visit_end, visit_start, visit_start],
                "code": ["VISIT/IP", "VISIT/IP", "DIAGNOSIS/flu", "LAB/glucose"],
                "visit_id": [100, 100, 100, 100],
                "workflow_stage": ["start", "end", None, None],
            }
        )
        mover = VisitEndTimeMover(VisitEndTimeConfig())
        result = mover(df)
        diag = result.filter(pl.col("code") == "DIAGNOSIS/flu")
        assert diag["time"][0] == visit_end
        lab = result.filter(pl.col("code") == "LAB/glucose")
        assert lab["time"][0] == visit_start
