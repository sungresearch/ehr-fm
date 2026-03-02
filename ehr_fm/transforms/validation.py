from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class VisitEndTimeConfig(BaseModel):
    """Configuration for moving event times to visit end.

    Uses workflow_stage column to identify visit start/end events.
    """

    code_prefixes: list[str] = Field(
        default_factory=lambda: ["DIAGNOSIS/", "PROCEDURE/", "OBSERVATION/"],
        description="Code prefixes for events to move to visit end time",
    )
    check_description: bool = Field(
        default=True,
        description="Also check description field for prefixes",
    )

    # Visit identification
    visit_code_pattern: str = Field(
        default="VISIT",
        description="Pattern to identify visit events in codes",
    )
    workflow_stage_start: str = Field(
        default="start",
        description="Value in workflow_stage column indicating visit start (case-insensitive)",
    )
    workflow_stage_end: str = Field(
        default="end",
        description="Value in workflow_stage column indicating visit end (case-insensitive)",
    )

    @model_validator(mode="after")
    def validate_visit_end_time_config(self) -> Self:
        if not self.code_prefixes:
            raise ValueError("Must provide at least one code prefix")
        return self


class TransformConfig(BaseModel):
    """Configuration for MEDS transformations."""

    # Active transforms
    move_to_visit_end: bool = Field(default=False)
    convert_to_meds_reader: bool = Field(default=True)
    verify_meds_reader: bool = Field(default=True)

    # Active configs
    visit_end_time_config: VisitEndTimeConfig | None = Field(default=None)
    meds_reader_output_dir: Path | None = Field(default=None)

    @model_validator(mode="after")
    def validate_transform_config(self) -> Self:
        if self.move_to_visit_end and not self.visit_end_time_config:
            self.visit_end_time_config = VisitEndTimeConfig()

        return self


def validate_transform_config(config: str | Path | dict | BaseModel) -> TransformConfig:
    """Validate transform configuration."""
    from ..validation import validate_config

    return validate_config(config, TransformConfig)
