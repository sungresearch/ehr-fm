"""Joint token builder for combined token string generation.

Provides JointTokenBuilder class that creates combined token strings
(e.g., "glucose/Q:3/TXT:positive/STAGE:inpatient") from event attributes.

This class is shared between:
- JointVocab: Uses build_token() to create keys for frequency counting during training
- JointPolicy: Uses build_token() to emit token strings during pretokenization

This ensures consistent token format across vocabulary training and pretokenization.
"""

from dataclasses import dataclass
from typing import Any

from .quantiles import assign_quantile_bucket
from .text import normalize_text
from .tokens import stage_token, stage_unk_token


@dataclass
class JointConfig:
    """Configuration for joint tokenization.

    Controls which attribute tokens are concatenated and how codes are processed.

    Attributes:
        emit_quantiles: Whether to append Q:* for numeric values
        emit_text: Whether to append TXT:* for text values
        emit_stage: Whether to append STAGE:* for workflow_stage
        num_quantiles: Number of quantile buckets (default 10)
        remove_prefixes: Whether to strip code prefixes for base token
        separator: Character used to join token parts
    """

    emit_quantiles: bool = True
    emit_text: bool = True
    emit_stage: bool = True
    num_quantiles: int = 10
    remove_prefixes: bool = True
    separator: str = "/"


class JointTokenBuilder:
    """Builds combined token strings for joint mode.

    Creates single tokens that concatenate base code with attribute tokens:
    - glucose/Q:3
    - result/TXT:positive
    - glucose/Q:3/STAGE:inpatient

    Used by both JointVocab (training) and JointPolicy (pretokenization)
    to ensure consistent token format.

    Args:
        config: JointConfig with emission settings
        quantile_breaks: {code: [break1, break2, ...]} for bucket assignment
        known_stages: Set of known stage values (for UNK detection)
    """

    def __init__(
        self,
        config: JointConfig,
        quantile_breaks: dict[str, list[float]] | None = None,
        known_stages: set[str] | None = None,
    ):
        self.config = config
        self.quantile_breaks = quantile_breaks or {}
        self.known_stages = {s.lower() for s in (known_stages or set())}

    def _extract_base_code(self, code: str) -> str:
        """Extract base code from full code string.

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

    def build_token(
        self,
        code: str,
        numeric_value: float | None = None,
        text_value: str | None = None,
        workflow_stage: str | None = None,
    ) -> str | None:
        """Build combined token string from event attributes.

        Concatenates base code with applicable attribute tokens:
        - base_code
        - base_code/Q:<k> (if numeric_value and emit_quantiles)
        - base_code/TXT:<norm> (if text_value and emit_text)
        - base_code/STAGE:<val> (if workflow_stage and emit_stage)
        - base_code/Q:<k>/TXT:<norm>/STAGE:<val> (all attributes)

        Args:
            code: Event code string
            numeric_value: Numeric value for quantile assignment
            text_value: Text value for normalization
            workflow_stage: Workflow stage value

        Returns:
            Combined token string, or None if code is empty
        """
        if not code:
            return None

        parts = [self._extract_base_code(code)]

        # Quantile part (if applicable)
        if numeric_value is not None and self.config.emit_quantiles:
            q_token = assign_quantile_bucket(code, numeric_value, self.quantile_breaks)
            # Extract just the Q:k part (remove "Q:" prefix if needed for consistency)
            parts.append(q_token)

        # Text part (if applicable)
        if text_value is not None and self.config.emit_text:
            normalized = normalize_text(text_value)
            if normalized:
                parts.append(f"TXT:{normalized}")

        # Stage part (if applicable)
        if workflow_stage is not None and self.config.emit_stage:
            stage_lower = workflow_stage.lower()
            if self.known_stages and stage_lower not in self.known_stages:
                parts.append(stage_unk_token())
            else:
                parts.append(stage_token(workflow_stage))

        return self.config.separator.join(parts)

    def build_token_from_event(self, event: dict[str, Any]) -> str | None:
        """Build combined token string from event dict.

        Convenience method that extracts fields from event dict.

        Args:
            event: Event dict with keys:
                - code: str
                - numeric_value: float | None
                - text_value: str | None
                - workflow_stage: str | None (optional)

        Returns:
            Combined token string, or None if code is empty
        """
        return self.build_token(
            code=event.get("code", ""),
            numeric_value=event.get("numeric_value"),
            text_value=event.get("text_value"),
            workflow_stage=event.get("workflow_stage"),
        )

    def get_quantile_token(self, code: str, numeric_value: float) -> str:
        """Get standalone quantile token for a value.

        Args:
            code: Event code for break lookup
            numeric_value: Value to bucket

        Returns:
            Token string like "Q:3" or "Q:UNK"
        """
        return assign_quantile_bucket(code, numeric_value, self.quantile_breaks)

    def get_stage_token(self, workflow_stage: str) -> str:
        """Get standalone stage token for a value.

        Args:
            workflow_stage: Stage value

        Returns:
            Token string like "STAGE:inpatient" or "STAGE:UNK"
        """
        stage_lower = workflow_stage.lower()
        if self.known_stages and stage_lower not in self.known_stages:
            return stage_unk_token()
        return stage_token(workflow_stage)

    def get_text_token(self, text_value: str) -> str | None:
        """Get standalone text token for a value.

        Args:
            text_value: Text value to normalize

        Returns:
            Token string like "TXT:positive", or None if normalizes to empty
        """
        normalized = normalize_text(text_value)
        if normalized:
            return f"TXT:{normalized}"
        return None
