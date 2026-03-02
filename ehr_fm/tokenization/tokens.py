"""Token string builders for factorized tokenization.

Provides canonical token string formats:
- Quantiles: Q:<k> and Q:UNK
- Stage: STAGE:<value> and STAGE:UNK
- Text: TXT:<normalized>

These builders ensure consistent token naming across vocabulary training
and pretokenization.
"""


def quantile_token(bucket: int) -> str:
    """Return e.g. ``"Q:3"`` for a 1-indexed bucket number."""
    return f"Q:{bucket}"


def quantile_unk_token() -> str:
    return "Q:UNK"


def stage_token(stage: str) -> str:
    """Return e.g. ``"STAGE:taken"`` (lowercased)."""
    return f"STAGE:{stage.lower()}"


def stage_unk_token() -> str:
    return "STAGE:UNK"


def text_token(normalized_text: str) -> str:
    """Return e.g. ``"TXT:oral"``."""
    return f"TXT:{normalized_text}"
