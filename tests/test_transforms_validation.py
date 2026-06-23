import pytest
from pydantic import ValidationError

from ehr_fm.transforms.validation import (
    TransformConfig,
    VisitEndTimeConfig,
    validate_transform_config,
)


class TestVisitEndTimeConfig:
    def test_custom_prefixes(self):
        cfg = VisitEndTimeConfig(code_prefixes=["LAB/", "CUSTOM/"])
        assert cfg.code_prefixes == ["LAB/", "CUSTOM/"]

    def test_empty_prefixes_raises(self):
        with pytest.raises(ValidationError, match="at least one code prefix"):
            VisitEndTimeConfig(code_prefixes=[])


class TestTransformConfig:
    def test_auto_creates_visit_end_config(self):
        cfg = TransformConfig(move_to_visit_end=True)
        assert cfg.visit_end_time_config is not None
        assert isinstance(cfg.visit_end_time_config, VisitEndTimeConfig)

    def test_respects_explicit_visit_end_config(self):
        explicit = VisitEndTimeConfig(code_prefixes=["CUSTOM/"])
        cfg = TransformConfig(move_to_visit_end=True, visit_end_time_config=explicit)
        assert cfg.visit_end_time_config.code_prefixes == ["CUSTOM/"]


class TestValidateTransformConfig:
    def test_dict_input(self):
        result = validate_transform_config({"move_to_visit_end": True})
        assert isinstance(result, TransformConfig)
        assert result.move_to_visit_end is True
        assert result.visit_end_time_config is not None

    def test_config_passthrough(self):
        cfg = TransformConfig(move_to_visit_end=False)
        result = validate_transform_config(cfg)
        assert result is cfg
