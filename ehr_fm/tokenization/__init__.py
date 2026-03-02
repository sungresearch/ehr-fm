"""Tokenization policies and utilities for ehr-fm.

This module provides the policy abstraction for token emission,
shared by both vocabulary training and pretokenization.

Main exports:
- TokenizationMode: Literal type for "joint" or "factorized"
- FactorizedConfig: Configuration dataclass for factorized mode
- JointConfig: Configuration dataclass for joint mode
- TokenizationPolicy: Abstract base for policies
- JointPolicy: Current joint tokenization behavior
- FactorizedPolicy: New factorized tokenization with on-the-fly attributes
- JointTokenBuilder: Shared builder for combined token strings in joint mode

Token builders:
- quantile_token, quantile_unk_token: Q:* tokens
- stage_token, stage_unk_token: STAGE:* tokens
- text_token: TXT:* tokens

Utilities:
- assign_quantile_bucket: Map numeric value to Q:* token
- normalize_text, make_text_token: Text normalization
"""

from .joint import JointConfig, JointTokenBuilder
from .policy import FactorizedPolicy, JointPolicy, TokenizationPolicy
from .quantiles import assign_quantile_bucket
from .text import make_text_token, normalize_text
from .tokens import (
    quantile_token,
    quantile_unk_token,
    stage_token,
    stage_unk_token,
    text_token,
)
from .types import FactorizedConfig, TokenizationMode

__all__ = [
    # Types
    "TokenizationMode",
    "FactorizedConfig",
    "JointConfig",
    # Policies
    "TokenizationPolicy",
    "JointPolicy",
    "FactorizedPolicy",
    # Joint token builder (shared between vocab and policy)
    "JointTokenBuilder",
    # Token builders
    "quantile_token",
    "quantile_unk_token",
    "stage_token",
    "stage_unk_token",
    "text_token",
    # Utilities
    "assign_quantile_bucket",
    "normalize_text",
    "make_text_token",
]
