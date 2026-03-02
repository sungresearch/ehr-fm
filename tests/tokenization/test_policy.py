"""Tests for tokenization policies."""

import pytest

from ehr_fm.tokenization.joint import JointConfig
from ehr_fm.tokenization.policy import FactorizedPolicy, JointPolicy
from ehr_fm.tokenization.types import FactorizedConfig


class TestJointPolicy:
    """Test JointPolicy for joint tokenization.

    JointPolicy uses JointTokenBuilder to create combined token strings.
    Each event emits exactly ONE token that concatenates base code with attributes.
    E.g., "glucose/Q:3/TXT:fasting/STAGE:taken"
    """

    @pytest.fixture
    def default_config(self):
        """Default JointConfig with all emissions enabled."""
        return JointConfig(
            emit_quantiles=True,
            emit_text=True,
            emit_stage=True,
            num_quantiles=10,
            remove_prefixes=True,
            separator="/",
        )

    @pytest.fixture
    def sample_breaks(self):
        """Sample quantile breaks."""
        return {"LAB/glucose": [70.0, 85.0, 100.0, 120.0]}

    @pytest.fixture
    def sample_stages(self):
        """Sample known stages."""
        return {"order", "taken", "admin"}

    # Basic token emission tests
    def test_basic_code_no_attributes(self, default_config):
        """Code without attributes emits single base token."""
        policy = JointPolicy(config=default_config)
        event = {"code": "MEDS_BIRTH"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["MEDS_BIRTH"]

    def test_remove_prefixes_true(self, default_config):
        """Prefix removed from base code."""
        policy = JointPolicy(config=default_config)
        event = {"code": "LAB/glucose"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["glucose"]

    def test_remove_prefixes_false(self):
        """Prefix kept when remove_prefixes=False."""
        config = JointConfig(remove_prefixes=False)
        policy = JointPolicy(config=config)
        event = {"code": "LAB/glucose"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["LAB/glucose"]

    def test_empty_code(self, default_config):
        """Empty code returns empty list."""
        policy = JointPolicy(config=default_config)
        event = {"code": ""}
        assert policy.emit_token_strings(event) == []

    def test_missing_code(self, default_config):
        """Missing code returns empty list."""
        policy = JointPolicy(config=default_config)
        event = {}
        assert policy.emit_token_strings(event) == []

    def test_default_config_when_none(self):
        """JointPolicy uses default config when None provided."""
        policy = JointPolicy()
        event = {"code": "MEDS_BIRTH"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["MEDS_BIRTH"]

    # Combined token tests with attributes
    def test_code_with_numeric_value(self, default_config, sample_breaks):
        """Code + numeric → single combined token."""
        policy = JointPolicy(
            config=default_config,
            quantile_breaks=sample_breaks,
        )
        event = {"code": "LAB/glucose", "numeric_value": 90.0}
        tokens = policy.emit_token_strings(event)
        # 90 is between 85 and 100 → Q:3
        assert len(tokens) == 1
        assert tokens[0] == "glucose/Q:3"

    def test_code_with_text_value(self, default_config):
        """Code + text → single combined token."""
        policy = JointPolicy(config=default_config)
        event = {"code": "DRUG/metformin", "text_value": "oral"}
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 1
        assert tokens[0] == "metformin/TXT:oral"

    def test_code_with_stage(self, default_config, sample_stages):
        """Code + stage → single combined token."""
        policy = JointPolicy(config=default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "taken"}
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 1
        assert tokens[0] == "glucose/STAGE:taken"

    def test_code_with_all_attributes(self, default_config, sample_breaks, sample_stages):
        """Code + all attributes → single combined token."""
        policy = JointPolicy(
            config=default_config,
            quantile_breaks=sample_breaks,
            known_stages=sample_stages,
        )
        event = {
            "code": "LAB/glucose",
            "numeric_value": 90.0,
            "text_value": "fasting",
            "workflow_stage": "taken",
        }
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 1
        assert tokens[0] == "glucose/Q:3/TXT:fasting/STAGE:taken"

    # OOV / UNK handling
    def test_quantile_unk_for_unknown_code(self, default_config, sample_breaks):
        """Code not in quantile_breaks → Q:UNK appended."""
        policy = JointPolicy(
            config=default_config,
            quantile_breaks=sample_breaks,
        )
        event = {"code": "LAB/unknown", "numeric_value": 100.0}
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 1
        assert tokens[0] == "unknown/Q:UNK"

    def test_stage_unk_for_unknown_stage(self, default_config, sample_stages):
        """Unknown stage → STAGE:UNK appended."""
        policy = JointPolicy(
            config=default_config,
            known_stages=sample_stages,
        )
        event = {"code": "LAB/glucose", "workflow_stage": "unknown_stage"}
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 1
        assert tokens[0] == "glucose/STAGE:UNK"

    # Emission disabled tests
    def test_quantile_not_emitted_when_disabled(self, sample_breaks):
        """No quantile appended when emit_quantiles=False."""
        config = JointConfig(
            emit_quantiles=False,
            emit_text=False,
            emit_stage=False,
            remove_prefixes=True,
        )
        policy = JointPolicy(config=config, quantile_breaks=sample_breaks)
        event = {"code": "LAB/glucose", "numeric_value": 90.0}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["glucose"]  # No Q:3 appended

    def test_text_not_emitted_when_disabled(self):
        """No text appended when emit_text=False."""
        config = JointConfig(
            emit_quantiles=False,
            emit_text=False,
            emit_stage=False,
            remove_prefixes=True,
        )
        policy = JointPolicy(config=config)
        event = {"code": "DRUG/metformin", "text_value": "oral"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["metformin"]  # No TXT:oral appended

    def test_stage_not_emitted_when_disabled(self, sample_stages):
        """No stage appended when emit_stage=False."""
        config = JointConfig(
            emit_quantiles=False,
            emit_text=False,
            emit_stage=False,
            remove_prefixes=True,
        )
        policy = JointPolicy(config=config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "taken"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["glucose"]  # No STAGE:taken appended

    # Edge cases
    def test_text_dropped_when_normalizes_empty(self, default_config):
        """Text not appended when it normalizes to empty."""
        policy = JointPolicy(config=default_config)
        event = {"code": "DRUG/metformin", "text_value": "@#$"}
        tokens = policy.emit_token_strings(event)
        assert tokens == ["metformin"]  # No TXT appended

    def test_null_attributes_ignored(self, default_config, sample_breaks, sample_stages):
        """None values for attributes are ignored."""
        policy = JointPolicy(
            config=default_config,
            quantile_breaks=sample_breaks,
            known_stages=sample_stages,
        )
        event = {
            "code": "LAB/glucose",
            "numeric_value": None,
            "text_value": None,
            "workflow_stage": None,
        }
        tokens = policy.emit_token_strings(event)
        assert tokens == ["glucose"]  # Just base code

    def test_stage_case_insensitive_matching(self, default_config, sample_stages):
        """Stage matching is case-insensitive."""
        policy = JointPolicy(config=default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "TAKEN"}
        tokens = policy.emit_token_strings(event)
        # Output uses original case, but matching is lowercase
        assert len(tokens) == 1
        assert "STAGE:taken" in tokens[0]


class TestFactorizedPolicy:
    """Test FactorizedPolicy for factorized tokenization."""

    @pytest.fixture
    def default_config(self):
        """Default factorized config."""
        return FactorizedConfig(
            emit_quantiles=True,
            emit_text=True,
            emit_stage=True,
            num_quantiles=10,
            remove_prefixes=True,
            separator="/",
        )

    @pytest.fixture
    def sample_breaks(self):
        """Sample quantile breaks."""
        return {"LAB/glucose": [70.0, 85.0, 100.0, 120.0]}

    @pytest.fixture
    def sample_stages(self):
        """Sample known stages."""
        return {"order", "taken", "admin"}

    # Base token extraction tests
    def test_base_token_extraction(self, default_config):
        """Base token extracted from code."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "LAB/glucose"}
        tokens = policy.emit_token_strings(event)
        assert tokens[0] == "glucose"

    def test_base_token_no_separator(self, default_config):
        """Code without separator returned as-is."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "MEDS_BIRTH"}
        tokens = policy.emit_token_strings(event)
        assert tokens[0] == "MEDS_BIRTH"

    def test_base_token_empty_after_prefix(self, default_config):
        """Code like 'LAB/' returns 'LAB/'."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "LAB/"}
        tokens = policy.emit_token_strings(event)
        # Empty after split → return original code
        assert tokens[0] == "LAB/"

    # Quantile emission tests
    def test_quantile_emitted_when_numeric_present(self, default_config, sample_breaks):
        """Q:k token emitted for numeric_value."""
        policy = FactorizedPolicy(default_config, sample_breaks)
        event = {"code": "LAB/glucose", "numeric_value": 90.0}
        tokens = policy.emit_token_strings(event)
        assert "Q:3" in tokens  # 90 is between 85 and 100

    def test_quantile_not_emitted_when_disabled(self, sample_breaks):
        """No Q token when emit_quantiles=False."""
        config = FactorizedConfig(emit_quantiles=False, emit_text=False, emit_stage=False)
        policy = FactorizedPolicy(config, sample_breaks)
        event = {"code": "LAB/glucose", "numeric_value": 90.0}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("Q:") for t in tokens)

    def test_quantile_not_emitted_when_null(self, default_config, sample_breaks):
        """No Q token when numeric_value is None."""
        policy = FactorizedPolicy(default_config, sample_breaks)
        event = {"code": "LAB/glucose", "numeric_value": None}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("Q:") for t in tokens)

    def test_quantile_unk_for_missing_code(self, default_config, sample_breaks):
        """Q:UNK when code not in quantile_breaks."""
        policy = FactorizedPolicy(default_config, sample_breaks)
        event = {"code": "LAB/unknown", "numeric_value": 100.0}
        tokens = policy.emit_token_strings(event)
        assert "Q:UNK" in tokens

    # Stage emission tests
    def test_stage_emitted_when_present(self, default_config, sample_stages):
        """STAGE:value token emitted."""
        policy = FactorizedPolicy(default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "taken"}
        tokens = policy.emit_token_strings(event)
        assert "STAGE:taken" in tokens

    def test_stage_not_emitted_when_disabled(self, sample_stages):
        """No STAGE token when emit_stage=False."""
        config = FactorizedConfig(emit_quantiles=False, emit_text=False, emit_stage=False)
        policy = FactorizedPolicy(config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "taken"}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("STAGE:") for t in tokens)

    def test_stage_not_emitted_when_null(self, default_config, sample_stages):
        """No STAGE token when workflow_stage is None."""
        policy = FactorizedPolicy(default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": None}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("STAGE:") for t in tokens)

    def test_stage_unk_for_unknown_value(self, default_config, sample_stages):
        """STAGE:UNK when stage not in known_stages."""
        policy = FactorizedPolicy(default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "unknown_stage"}
        tokens = policy.emit_token_strings(event)
        assert "STAGE:UNK" in tokens

    def test_stage_case_insensitive(self, default_config, sample_stages):
        """Stage matching is case-insensitive."""
        policy = FactorizedPolicy(default_config, known_stages=sample_stages)
        event = {"code": "LAB/glucose", "workflow_stage": "TAKEN"}
        tokens = policy.emit_token_strings(event)
        assert "STAGE:taken" in tokens

    # Text emission tests
    def test_text_emitted_when_present(self, default_config):
        """TXT:norm token emitted."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "DRUG/metformin", "text_value": "oral"}
        tokens = policy.emit_token_strings(event)
        assert "TXT:oral" in tokens

    def test_text_not_emitted_when_disabled(self):
        """No TXT token when emit_text=False."""
        config = FactorizedConfig(emit_quantiles=False, emit_text=False, emit_stage=False)
        policy = FactorizedPolicy(config)
        event = {"code": "DRUG/metformin", "text_value": "oral"}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("TXT:") for t in tokens)

    def test_text_not_emitted_when_null(self, default_config):
        """No TXT token when text_value is None."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "DRUG/metformin", "text_value": None}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("TXT:") for t in tokens)

    def test_text_dropped_when_normalizes_empty(self, default_config):
        """No TXT token when text normalizes to empty string."""
        policy = FactorizedPolicy(default_config)
        event = {"code": "DRUG/metformin", "text_value": "@#$"}
        tokens = policy.emit_token_strings(event)
        assert not any(t.startswith("TXT:") for t in tokens)

    # Combined tests
    def test_full_event_all_attributes(self, default_config, sample_breaks, sample_stages):
        """Event with all attributes emits [base, Q, TXT, STAGE]."""
        policy = FactorizedPolicy(default_config, sample_breaks, sample_stages)
        event = {
            "code": "LAB/glucose",
            "numeric_value": 90.0,
            "text_value": "fasting",
            "workflow_stage": "taken",
        }
        tokens = policy.emit_token_strings(event)
        assert len(tokens) == 4
        assert tokens[0] == "glucose"
        assert "Q:3" in tokens
        assert "TXT:fasting" in tokens
        assert "STAGE:taken" in tokens

    def test_token_order(self, default_config, sample_breaks, sample_stages):
        """Tokens emitted in order: base, Q, TXT, STAGE."""
        policy = FactorizedPolicy(default_config, sample_breaks, sample_stages)
        event = {
            "code": "LAB/glucose",
            "numeric_value": 90.0,
            "text_value": "fasting",
            "workflow_stage": "taken",
        }
        tokens = policy.emit_token_strings(event)
        # Order: base, Q, TXT, STAGE
        assert tokens[0] == "glucose"
        assert tokens[1] == "Q:3"
        assert tokens[2] == "TXT:fasting"
        assert tokens[3] == "STAGE:taken"
