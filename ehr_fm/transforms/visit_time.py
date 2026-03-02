"""Visit end time transform for MEDS data.

This module provides functionality to move event timestamps from visit start
to visit end for specified code prefixes, addressing data leakage in OMOP-derived
MEDS data.
"""

import polars as pl

from ..logger import setup_logging
from .validation import VisitEndTimeConfig

logger = setup_logging(child_name="transforms.visit_time")


class VisitEndTimeMover:
    """Move event timestamps from visit start to visit end for specified prefixes.

    This transform addresses data leakage in OMOP-derived MEDS data where certain events
    (e.g., diagnosis, procedures, observations) that actually occurred at visit end are
    timestamped with visit start datetime by default.

    Uses the workflow_stage column to identify visit start/end events.

    The transform:
    1. Identifies visit start and end times from events with workflow_stage = "start"/"end"
    2. For events with specified code/description prefixes where event_time == visit_start_time
    3. Updates their timestamp to visit_end_time
    4. Ensures proper sorting by subject_id and time
    """

    def __init__(self, config: VisitEndTimeConfig):
        self.config = config

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        # Check required columns
        if "workflow_stage" not in df.columns:
            logger.error(
                "workflow_stage column not found in DataFrame. "
                "Cannot identify visit start/end events. Returning unchanged data."
            )
            return df.sort(["subject_id", "time"], maintain_order=True)

        if "visit_id" not in df.columns:
            logger.warning(
                "visit_id column not found in DataFrame. Cannot move events to visit end time. "
                "Returning unchanged data."
            )
            return df.sort(["subject_id", "time"], maintain_order=True)

        if "code" not in df.columns:
            logger.warning("code column not found in DataFrame. Returning unchanged data.")
            return df.sort(["subject_id", "time"], maintain_order=True)

        original_rows = len(df)
        logger.info(f"Processing {original_rows} events for visit end time adjustment...")

        visit_times = self._extract_visit_times(df)

        if visit_times is None or len(visit_times) == 0:
            logger.warning("No visit start/end events found. Returning unchanged data.")
            return df.sort(["subject_id", "time"], maintain_order=True)

        result_df = self._update_event_times(df, visit_times)
        result_df = result_df.sort(["subject_id", "time"], maintain_order=True)

        events_moved = self._count_events_moved(df, result_df)
        if events_moved > 0:
            logger.info(
                f"Moved {events_moved} events ({events_moved/original_rows*100:.2f}%) to visit end time"
            )
        else:
            logger.info("No events were moved to visit end time")

        return result_df

    def _extract_visit_times(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Return a DF with (visit_id, visit_start_time, visit_end_time), or None."""
        # Identify visit events by code pattern
        is_visit_event = pl.col("code").str.contains(self.config.visit_code_pattern)

        # Case-insensitive workflow_stage matching
        workflow_stage_lower = pl.col("workflow_stage").str.to_lowercase()

        # Identify visit start events: visit code + workflow_stage == "start" (case-insensitive)
        visit_start_filter = (
            is_visit_event
            & pl.col("workflow_stage").is_not_null()
            & (workflow_stage_lower == self.config.workflow_stage_start.lower())
        )

        # Identify visit end events: visit code + workflow_stage == "end" (case-insensitive)
        visit_end_filter = (
            is_visit_event
            & pl.col("workflow_stage").is_not_null()
            & (workflow_stage_lower == self.config.workflow_stage_end.lower())
        )

        # Extract visit start times - use min() for earliest time
        visit_starts = (
            df.filter(visit_start_filter & pl.col("visit_id").is_not_null())
            .group_by("visit_id")
            .agg(pl.col("time").min().alias("visit_start_time"))
        )

        # Extract visit end times - use max() for latest time
        visit_ends = (
            df.filter(visit_end_filter & pl.col("visit_id").is_not_null())
            .group_by("visit_id")
            .agg(pl.col("time").max().alias("visit_end_time"))
        )

        logger.info(f"Found {len(visit_starts)} visit start events, {len(visit_ends)} visit end events")

        if len(visit_starts) == 0 or len(visit_ends) == 0:
            return None

        # Join start and end times
        visit_times = visit_starts.join(visit_ends, on="visit_id", how="inner")

        logger.info(f"Found {len(visit_times)} visits with both start and end times")

        return visit_times

    def _update_event_times(self, df: pl.DataFrame, visit_times: pl.DataFrame) -> pl.DataFrame:
        # Create prefix matching condition for code
        code_matches = pl.lit(False)
        for prefix in self.config.code_prefixes:
            code_matches = code_matches | pl.col("code").str.starts_with(prefix)

        # Create prefix matching condition for description if enabled
        if self.config.check_description and "description" in df.columns:
            description_matches = pl.lit(False)
            for prefix in self.config.code_prefixes:
                description_matches = description_matches | pl.col("description").str.starts_with(prefix)
            prefix_matches = code_matches | description_matches
        else:
            prefix_matches = code_matches

        # Join with visit times
        df_with_visit_times = df.join(visit_times, on="visit_id", how="left", maintain_order="left")

        # Update time for events that:
        # 1. Match the prefix criteria (code OR description)
        # 2. Have a visit_start_time and visit_end_time
        # 3. Current event time equals visit_start_time (to be safe)
        # 4. Have a non-null visit_id
        should_move = (
            prefix_matches
            & pl.col("visit_id").is_not_null()
            & pl.col("visit_start_time").is_not_null()
            & pl.col("visit_end_time").is_not_null()
            & (pl.col("time") == pl.col("visit_start_time"))
        )

        # Create updated dataframe
        result_df = df_with_visit_times.with_columns(
            pl.when(should_move).then(pl.col("visit_end_time")).otherwise(pl.col("time")).alias("time")
        )

        # Drop the temporary columns
        result_df = result_df.drop(["visit_start_time", "visit_end_time"])

        return result_df

    def _count_events_moved(self, original_df: pl.DataFrame, result_df: pl.DataFrame) -> int:
        original_with_idx = original_df.with_row_index("_idx")
        result_with_idx = result_df.with_row_index("_idx")

        comparison = original_with_idx.join(result_with_idx, on="_idx", how="inner")

        if "time" in comparison.columns and "time_right" in comparison.columns:
            moved = comparison.filter(pl.col("time") != pl.col("time_right")).height
            return moved
        elif "time" in comparison.columns:
            return 0
        else:
            return 0
