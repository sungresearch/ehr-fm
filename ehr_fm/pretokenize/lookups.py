"""Vocab-index lookups and the per-event tokenization bridge used at pretokenize time.

These are pure functions over an already-built vocabulary: they resolve emitted
token *strings* to integer vocab indices and enforce the no-orphans / vocab-cutoff
rules. They produce token ids, not embeddings.
"""

from typing import Any

from ehr_fm.tokenization import TokenizationPolicy


def _build_token_string_lookup(vocab: list[dict[str, Any]]) -> dict[str, int]:
    """Build a lookup from token string to vocab index for factorized tokenization.

    Maps:
    - code entries: code_string → index
    - quantile entries: label (Q:*) → index
    - stage entries: label (STAGE:*) → index
    - text entries: label (TXT:*) → index
    - interval entries: label → index
    - demographic entries: label → index
    """
    lookup: dict[str, int] = {}
    for i, entry in enumerate(vocab):
        entry_type = entry.get("type")
        if entry_type == "code":
            code_string = entry.get("code_string")
            if code_string:
                lookup[code_string] = i
        elif entry_type in ("quantile", "stage", "interval", "demographic"):
            label = entry.get("label")
            if label:
                lookup[label] = i
        elif entry_type == "text":
            # Text entries use TXT:<normalized> format
            text_string = entry.get("text_string")
            if text_string:
                lookup[f"TXT:{text_string}"] = i
        # numeric entries are not used in factorized mode (use Q:* instead)
    return lookup


def _build_interval_token_lookup(vocab: list[dict[str, Any]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, v in enumerate(vocab):
        if v.get("type") == "interval":
            lbl = v.get("label")
            if isinstance(lbl, str):
                lookup[lbl] = i
    return lookup


def _tokenize_event_with_policy(
    event: dict[str, Any],
    policy: TokenizationPolicy,
    token_lookup: dict[str, int],
    vocab_size: int,
) -> list[int]:
    """Tokenize a single event using a tokenization policy.

    Works for both joint and factorized modes:
    - JointPolicy: emits single combined token like "4096101/Q:7"
    - FactorizedPolicy: emits multiple tokens like ["4096101", "Q:7"]

    Returns list of token IDs. Empty list if base token is OOV (no-orphans invariant).

    Args:
        event: Event dict with code, numeric_value, text_value, workflow_stage
        policy: TokenizationPolicy instance (JointPolicy or FactorizedPolicy)
        token_lookup: token_string → vocab_index mapping
        vocab_size: Maximum token ID (for cutoff filtering)

    Returns:
        List of valid token IDs for this event
    """
    token_strings = policy.emit_token_strings(event)
    if not token_strings:
        return []

    # First token is always the base token - check for OOV
    base_token_str = token_strings[0]
    base_tok_id = token_lookup.get(base_token_str)

    # No-orphans invariant: if base token is OOV, skip all tokens for this event
    if base_tok_id is None or base_tok_id >= vocab_size:
        return []

    # Collect valid token IDs
    valid_tokens = [base_tok_id]
    for token_str in token_strings[1:]:
        tok_id = token_lookup.get(token_str)
        if tok_id is not None and tok_id < vocab_size:
            valid_tokens.append(tok_id)

    return valid_tokens
