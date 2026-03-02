"""Tokenization policies for joint and factorized modes.

Policies define how events are converted to token strings.
The same policy is used by both vocabulary training and pretokenization
to ensure consistent token emission.

Design:
- TokenizationPolicy: Abstract base defining the emit_token_strings interface
- JointPolicy: Builds combined tokens using JointTokenBuilder (e.g., "glucose/Q:3")
- FactorizedPolicy: Emits separate tokens on-the-fly (e.g., ["glucose", "Q:3"])

"""

from abc import ABC, abstractmethod
from typing import Any

from .joint import JointConfig, JointTokenBuilder
from .quantiles import assign_quantile_bucket
from .text import make_text_token
from .tokens import stage_token, stage_unk_token
from .types import FactorizedConfig


class TokenizationPolicy(ABC):
    """Base class for tokenization policies.

    A policy defines how to convert an event dict into token strings.
    Token strings are then looked up in the vocabulary to get IDs.
    """

    @abstractmethod
    def emit_token_strings(self, event: dict[str, Any]) -> list[str]:
        """Emit token strings for an event (flat mode).

        Args:
            event: Event dict with keys:
                - code: str
                - numeric_value: float | None
                - text_value: str | None
                - workflow_stage: str | None (optional)

        Returns:
            List of token strings to look up in vocabulary.
            First token is always the base/code token.
        """
        ...


class JointPolicy(TokenizationPolicy):
    """Joint tokenization: event → single combined token.

    Builds combined tokens that concatenate base code with attribute tokens:
    - "glucose" (base code only)
    - "glucose/Q:3" (code + quantile)
    - "glucose/Q:3/TXT:positive" (code + quantile + text)
    - "glucose/Q:3/STAGE:inpatient" (code + quantile + stage)

    Uses JointTokenBuilder to ensure consistent token format between
    vocabulary training (JointVocab) and pretokenization (this policy).

    Args:
        config: JointConfig with emission settings
        quantile_breaks: {code: [break1, break2, ...]} for bucket assignment
        known_stages: Set of known stage values (for UNK detection)
    """

    def __init__(
        self,
        config: JointConfig | None = None,
        quantile_breaks: dict[str, list[float]] | None = None,
        known_stages: set[str] | None = None,
        token_lookup: dict[str, int] | None = None,
    ):
        if config is None:
            config = JointConfig()
        self.config = config
        self.builder = JointTokenBuilder(
            config=config,
            quantile_breaks=quantile_breaks,
            known_stages=known_stages,
        )
        self.token_lookup = token_lookup or {}

    def emit_token_strings(self, event: dict[str, Any]) -> list[str]:
        """Emit token strings for an event.

        Returns:
            List containing the combined token string, or empty if None.
        """
        combined = self.builder.build_token_from_event(event)
        if combined is None:
            return []
        return [combined]


class FactorizedPolicy(TokenizationPolicy):
    """Factorized tokenization: base code + attribute tokens.

    Emits tokens on-the-fly without requiring ETL expansion:
    - Base token: code with prefix removed
    - Q:<k>: quantile token (if numeric_value present and emit_quantiles=True)
    - TXT:<norm>: text token (if text_value present and emit_text=True)
    - STAGE:<value>: stage token (if workflow_stage present and emit_stage=True)

    Args:
        config: FactorizedConfig with emission settings
        quantile_breaks: {code: [break1, break2, ...]} for bucket assignment
        known_stages: Set of known stage values (for UNK detection)
    """

    def __init__(
        self,
        config: FactorizedConfig,
        quantile_breaks: dict[str, list[float]] | None = None,
        known_stages: set[str] | None = None,
        token_lookup: dict[str, int] | None = None,
    ):
        self.config = config
        self.quantile_breaks = quantile_breaks or {}
        self.known_stages = {s.lower() for s in (known_stages or set())}
        self.token_lookup = token_lookup or {}

    def _extract_base_token(self, code: str) -> str:
        """Extract base token from code.

        If remove_prefixes=True, returns second part after split.
        E.g., "LAB/glucose" → "glucose"
        """
        if not self.config.remove_prefixes:
            return code

        if self.config.separator not in code:
            return code

        parts = code.split(self.config.separator)
        # Return second part (first non-prefix part)
        if len(parts) > 1 and parts[1]:
            return parts[1]
        return code

    def _get_attribute_tokens(self, event: dict[str, Any]) -> list[str]:
        """Get attribute tokens for an event (Q:*, TXT:*, STAGE:*)."""
        code = event.get("code", "")
        tokens = []

        # Quantile token
        if self.config.emit_quantiles:
            numeric_value = event.get("numeric_value")
            if numeric_value is not None:
                q_token = assign_quantile_bucket(code, numeric_value, self.quantile_breaks)
                tokens.append(q_token)

        # Text token
        if self.config.emit_text:
            text_value = event.get("text_value")
            if text_value is not None:
                txt_token = make_text_token(text_value)
                if txt_token is not None:
                    tokens.append(txt_token)

        # Stage token
        if self.config.emit_stage:
            workflow_stage = event.get("workflow_stage")
            if workflow_stage is not None:
                stage_lower = workflow_stage.lower()
                if stage_lower in self.known_stages:
                    tokens.append(stage_token(workflow_stage))
                else:
                    tokens.append(stage_unk_token())

        return tokens

    def emit_token_strings(self, event: dict[str, Any]) -> list[str]:
        """Emit base token + attribute tokens for an event.

        Token order: [base, Q:*, TXT:*, STAGE:*]
        """
        code = event.get("code", "")
        if not code:
            return []

        base = self._extract_base_token(code)
        tokens = [base]
        tokens.extend(self._get_attribute_tokens(event))
        return tokens
