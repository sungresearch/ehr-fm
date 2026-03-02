"""Type definitions for tokenization configuration.

This module defines the core types and configuration dataclasses for
joint and factorized tokenization modes.
"""

from dataclasses import dataclass
from typing import Literal

TokenizationMode = Literal["joint", "factorized"]


@dataclass
class FactorizedConfig:
    """Configuration for factorized tokenization.

    Controls which attribute tokens are emitted and how codes are processed.

    Attributes:
        emit_quantiles: Whether to emit Q:* tokens for numeric values
        emit_text: Whether to emit TXT:* tokens for text values
        emit_stage: Whether to emit STAGE:* tokens for workflow_stage
        num_quantiles: Number of quantile buckets (default 10)
        remove_prefixes: Whether to strip code prefixes for base token
        separator: Code separator character
    """

    emit_quantiles: bool = True
    emit_text: bool = True
    emit_stage: bool = True
    num_quantiles: int = 10
    remove_prefixes: bool = True
    separator: str = "/"
