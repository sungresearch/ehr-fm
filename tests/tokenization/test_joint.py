"""Tests for JointTokenBuilder and JointConfig."""

import pytest

from ehr_fm.tokenization.joint import JointConfig, JointTokenBuilder


class TestJointConfig:
    """Test JointConfig dataclass defaults and values."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = JointConfig()
        assert config.emit_quantiles is True
        assert config.emit_text is True
        assert config.emit_stage is True
        assert config.num_quantiles == 10
        assert config.remove_prefixes is True
        assert config.separator == "/"

    def test_custom_values(self):
        """Config accepts custom values."""
        config = JointConfig(
            emit_quantiles=False,
            emit_text=False,
            emit_stage=False,
            num_quantiles=5,
            remove_prefixes=False,
            separator="-",
        )
        assert config.emit_quantiles is False
        assert config.emit_text is False
        assert config.emit_stage is False
        assert config.num_quantiles == 5
        assert config.remove_prefixes is False
        assert config.separator == "-"


class TestJointTokenBuilderBasicTokens:
    """Test basic token building without attributes."""

    @pytest.fixture
    def default_builder(self):
        """Builder with default config."""
        return JointTokenBuilder(config=JointConfig())

    def test_simple_code(self, default_builder):
        """Simple code without separator returns as-is."""
        result = default_builder.build_token("MEDS_BIRTH")
        assert result == "MEDS_BIRTH"

    def test_code_with_prefix_removed(self, default_builder):
        """Code with prefix → prefix removed."""
        result = default_builder.build_token("LAB/glucose")
        assert result == "glucose"

    def test_code_with_prefix_kept(self):
        """Prefix kept when remove_prefixes=False."""
        builder = JointTokenBuilder(config=JointConfig(remove_prefixes=False))
        result = builder.build_token("LAB/glucose")
        assert result == "LAB/glucose"

    def test_empty_code(self, default_builder):
        """Empty code returns None."""
        result = default_builder.build_token("")
        assert result is None

    def test_code_with_multiple_separators(self, default_builder):
        """Multiple separators → only first prefix removed."""
        result = default_builder.build_token("LAB/blood/glucose")
        assert result == "blood"  # Returns second part (first non-prefix)

    def test_code_ending_with_separator(self, default_builder):
        """Trailing separator → returns original if second part is empty."""
        result = default_builder.build_token("LAB/")
        assert result == "LAB/"


class TestJointTokenBuilderQuantiles:
    """Test quantile token appending."""

    @pytest.fixture
    def sample_breaks(self):
        """Sample quantile breaks for LAB/glucose."""
        return {"LAB/glucose": [70.0, 100.0, 130.0]}  # 4 buckets

    @pytest.fixture
    def builder_with_breaks(self, sample_breaks):
        """Builder with quantile breaks."""
        return JointTokenBuilder(
            config=JointConfig(emit_quantiles=True),
            quantile_breaks=sample_breaks,
        )

    def test_quantile_appended(self, builder_with_breaks):
        """Numeric value → quantile token appended."""
        result = builder_with_breaks.build_token("LAB/glucose", numeric_value=85.0)
        # 85 >= 70 → Q:2
        assert result == "glucose/Q:2"

    def test_quantile_first_bucket(self, builder_with_breaks):
        """Value below first break → Q:1."""
        result = builder_with_breaks.build_token("LAB/glucose", numeric_value=50.0)
        assert result == "glucose/Q:1"

    def test_quantile_last_bucket(self, builder_with_breaks):
        """Value above all breaks → last bucket."""
        result = builder_with_breaks.build_token("LAB/glucose", numeric_value=150.0)
        assert result == "glucose/Q:4"

    def test_quantile_exact_break(self, builder_with_breaks):
        """Value exactly on break → next bucket (>= comparison)."""
        result = builder_with_breaks.build_token("LAB/glucose", numeric_value=100.0)
        assert result == "glucose/Q:3"

    def test_quantile_unk_for_unknown_code(self, builder_with_breaks):
        """Unknown code → Q:UNK."""
        result = builder_with_breaks.build_token("LAB/unknown", numeric_value=100.0)
        assert result == "unknown/Q:UNK"

    def test_quantile_disabled(self, sample_breaks):
        """No quantile when emit_quantiles=False."""
        builder = JointTokenBuilder(
            config=JointConfig(emit_quantiles=False),
            quantile_breaks=sample_breaks,
        )
        result = builder.build_token("LAB/glucose", numeric_value=85.0)
        assert result == "glucose"  # No /Q:2 appended

    def test_quantile_none_value(self, builder_with_breaks):
        """None numeric_value → no quantile appended."""
        result = builder_with_breaks.build_token("LAB/glucose", numeric_value=None)
        assert result == "glucose"


class TestJointTokenBuilderText:
    """Test text token appending."""

    @pytest.fixture
    def builder_with_text(self):
        """Builder with text emission enabled."""
        return JointTokenBuilder(config=JointConfig(emit_text=True))

    def test_text_appended(self, builder_with_text):
        """Text value → TXT token appended."""
        result = builder_with_text.build_token("DRUG/metformin", text_value="oral")
        assert result == "metformin/TXT:oral"

    def test_text_normalized(self, builder_with_text):
        """Text is normalized (lowercase, spaces → underscores)."""
        result = builder_with_text.build_token("DRUG/metformin", text_value="Oral Tablet")
        assert result == "metformin/TXT:oral_tablet"

    def test_text_special_chars_removed(self, builder_with_text):
        """Special characters removed from text."""
        result = builder_with_text.build_token("DRUG/metformin", text_value="oral (500mg)")
        assert result == "metformin/TXT:oral_500mg"

    def test_text_empty_after_normalize(self, builder_with_text):
        """Text that normalizes to empty is not appended."""
        result = builder_with_text.build_token("DRUG/metformin", text_value="@#$%")
        assert result == "metformin"

    def test_text_disabled(self):
        """No text when emit_text=False."""
        builder = JointTokenBuilder(config=JointConfig(emit_text=False))
        result = builder.build_token("DRUG/metformin", text_value="oral")
        assert result == "metformin"

    def test_text_none_value(self, builder_with_text):
        """None text_value → no text appended."""
        result = builder_with_text.build_token("DRUG/metformin", text_value=None)
        assert result == "metformin"


class TestJointTokenBuilderStage:
    """Test stage token appending."""

    @pytest.fixture
    def builder_with_stages(self):
        """Builder with known stages."""
        return JointTokenBuilder(
            config=JointConfig(emit_stage=True),
            known_stages={"order", "taken", "admin"},
        )

    def test_stage_appended(self, builder_with_stages):
        """Known stage → STAGE token appended."""
        result = builder_with_stages.build_token("LAB/glucose", workflow_stage="taken")
        assert result == "glucose/STAGE:taken"

    def test_stage_case_insensitive(self, builder_with_stages):
        """Stage matching is case-insensitive."""
        result = builder_with_stages.build_token("LAB/glucose", workflow_stage="TAKEN")
        assert result == "glucose/STAGE:taken"

    def test_stage_unk_for_unknown(self, builder_with_stages):
        """Unknown stage → STAGE:UNK."""
        result = builder_with_stages.build_token("LAB/glucose", workflow_stage="unknown_stage")
        assert result == "glucose/STAGE:UNK"

    def test_stage_disabled(self):
        """No stage when emit_stage=False."""
        builder = JointTokenBuilder(
            config=JointConfig(emit_stage=False),
            known_stages={"taken"},
        )
        result = builder.build_token("LAB/glucose", workflow_stage="taken")
        assert result == "glucose"

    def test_stage_none_value(self, builder_with_stages):
        """None workflow_stage → no stage appended."""
        result = builder_with_stages.build_token("LAB/glucose", workflow_stage=None)
        assert result == "glucose"

    def test_stage_no_known_stages(self):
        """No known stages means all stages are valid."""
        builder = JointTokenBuilder(
            config=JointConfig(emit_stage=True),
            known_stages=None,  # Empty/None known_stages
        )
        result = builder.build_token("LAB/glucose", workflow_stage="any_stage")
        assert result == "glucose/STAGE:any_stage"


class TestJointTokenBuilderCombined:
    """Test combined token building with multiple attributes."""

    @pytest.fixture
    def full_builder(self):
        """Builder with all features enabled."""
        return JointTokenBuilder(
            config=JointConfig(
                emit_quantiles=True,
                emit_text=True,
                emit_stage=True,
                remove_prefixes=True,
            ),
            quantile_breaks={"LAB/glucose": [70.0, 100.0, 130.0]},
            known_stages={"taken", "order"},
        )

    def test_all_attributes_combined(self, full_builder):
        """All attributes → combined token in order."""
        result = full_builder.build_token(
            "LAB/glucose",
            numeric_value=85.0,
            text_value="fasting",
            workflow_stage="taken",
        )
        assert result == "glucose/Q:2/TXT:fasting/STAGE:taken"

    def test_quantile_and_text(self, full_builder):
        """Quantile + text combined."""
        result = full_builder.build_token(
            "LAB/glucose",
            numeric_value=85.0,
            text_value="fasting",
        )
        assert result == "glucose/Q:2/TXT:fasting"

    def test_quantile_and_stage(self, full_builder):
        """Quantile + stage combined."""
        result = full_builder.build_token(
            "LAB/glucose",
            numeric_value=85.0,
            workflow_stage="taken",
        )
        assert result == "glucose/Q:2/STAGE:taken"

    def test_text_and_stage(self, full_builder):
        """Text + stage combined (no numeric)."""
        result = full_builder.build_token(
            "DRUG/metformin",
            text_value="oral",
            workflow_stage="order",
        )
        assert result == "metformin/TXT:oral/STAGE:order"

    def test_custom_separator(self):
        """Custom separator used in combined token.

        Note: separator is used both for splitting (prefix removal) and joining.
        So with separator="|", code "LAB|glucose" would be split, but "LAB/glucose" would not.
        """
        builder = JointTokenBuilder(
            config=JointConfig(separator="|", remove_prefixes=True),
            quantile_breaks={"LAB|glucose": [100.0]},
        )
        result = builder.build_token("LAB|glucose", numeric_value=50.0)
        assert result == "glucose|Q:1"

    def test_custom_separator_join_only(self):
        """Custom separator only affects joining when code uses different separator."""
        builder = JointTokenBuilder(
            config=JointConfig(separator="|", remove_prefixes=False),
            quantile_breaks={"LAB/glucose": [100.0]},
        )
        result = builder.build_token("LAB/glucose", numeric_value=50.0)
        # Code kept as-is since remove_prefixes=False, joined with |
        assert result == "LAB/glucose|Q:1"


class TestJointTokenBuilderFromEvent:
    """Test build_token_from_event convenience method."""

    @pytest.fixture
    def builder(self):
        """Builder with all features."""
        return JointTokenBuilder(
            config=JointConfig(),
            quantile_breaks={"LAB/glucose": [100.0]},
            known_stages={"taken"},
        )

    def test_build_from_event_dict(self, builder):
        """Build token from event dict."""
        event = {
            "code": "LAB/glucose",
            "numeric_value": 50.0,
            "text_value": "fasting",
            "workflow_stage": "taken",
        }
        result = builder.build_token_from_event(event)
        assert result == "glucose/Q:1/TXT:fasting/STAGE:taken"

    def test_build_from_partial_event(self, builder):
        """Build from event with missing optional fields."""
        event = {"code": "LAB/glucose"}
        result = builder.build_token_from_event(event)
        assert result == "glucose"

    def test_build_from_empty_event(self, builder):
        """Build from event with no code returns None."""
        event = {}
        result = builder.build_token_from_event(event)
        assert result is None


class TestJointTokenBuilderHelperMethods:
    """Test standalone helper methods."""

    @pytest.fixture
    def builder(self):
        """Builder with config."""
        return JointTokenBuilder(
            config=JointConfig(),
            quantile_breaks={"LAB/glucose": [100.0]},
            known_stages={"taken"},
        )

    def test_get_quantile_token(self, builder):
        """get_quantile_token returns standalone quantile token."""
        result = builder.get_quantile_token("LAB/glucose", 50.0)
        assert result == "Q:1"

        result = builder.get_quantile_token("LAB/glucose", 150.0)
        assert result == "Q:2"

        result = builder.get_quantile_token("LAB/unknown", 100.0)
        assert result == "Q:UNK"

    def test_get_stage_token(self, builder):
        """get_stage_token returns standalone stage token."""
        result = builder.get_stage_token("taken")
        assert result == "STAGE:taken"

        result = builder.get_stage_token("unknown")
        assert result == "STAGE:UNK"

    def test_get_text_token(self, builder):
        """get_text_token returns standalone text token."""
        result = builder.get_text_token("oral")
        assert result == "TXT:oral"

        result = builder.get_text_token("Oral Tablet")
        assert result == "TXT:oral_tablet"

        result = builder.get_text_token("@#$")
        assert result is None
