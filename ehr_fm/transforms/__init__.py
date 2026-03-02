"""Optional MEDS preprocessing transforms.

Currently provides a single transform -- :class:`VisitEndTimeMover` -- which
fixes visit-end-time data leakage in OMOP-derived datasets (e.g. MIMIC) where
diagnosis, procedure, and observation events are timestamped at visit start
instead of visit end.

:class:`MEDSTransformPipeline` orchestrates reading a MEDS dataset, applying
transforms, copying source metadata, and optionally converting the result to
MEDS Reader format.

Typical usage is via the ``apply_meds_transforms`` CLI with
``--move-to-visit-end``.
"""

from .core import MEDSTransformPipeline
from .validation import TransformConfig, VisitEndTimeConfig, validate_transform_config
from .visit_time import VisitEndTimeMover

__all__ = [
    "MEDSTransformPipeline",
    "VisitEndTimeMover",
    "validate_transform_config",
    "TransformConfig",
    "VisitEndTimeConfig",
]
