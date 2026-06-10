"""Static vocabulary entry builders for interval, demographic, quantile, and stage tokens."""

from typing import Any

from ehr_fm.defaults import (
    DEFAULT_DEMOGRAPHIC_LABELS,
    DEFAULT_INTERVAL_LABELS,
    STAGE_UNK_LABEL,
    quantile_labels,
)


def _default_interval_vocab_entries() -> list[dict[str, Any]]:
    """Build interval token entries with stable labels and type 'interval'.

    Order matters: these are prepended to ensure low IDs and inclusion.
    Includes repeatable 6-month token label at the end.
    """
    return [{"type": "interval", "label": lbl, "weight": -1.0} for lbl in DEFAULT_INTERVAL_LABELS]


def prepend_interval_tokens(
    vocab_entries: list[dict[str, Any]], target_vocab_size: int | None = None
) -> list[dict[str, Any]]:
    """Prepend interval tokens to the ranked vocabulary, then truncate to original size.

    - Ensures interval tokens receive the lowest IDs and are never truncated out.
    - Avoids duplicates if intervals already exist.
    """
    intervals = _default_interval_vocab_entries()
    existing_labels = {v.get("label") for v in vocab_entries if v.get("type") == "interval"}
    to_add = [e for e in intervals if e["label"] not in existing_labels]

    combined = to_add + vocab_entries
    # Preserve or cap at configured vocab size if provided
    target_size = target_vocab_size if target_vocab_size is not None else len(vocab_entries)
    if len(combined) > target_size:
        return combined[:target_size]
    return combined


def _default_demographic_vocab_entries() -> list[dict[str, Any]]:
    """Build demographic token entries: sex, age buckets, first-event year buckets."""
    return [{"type": "demographic", "label": lbl, "weight": -1.0} for lbl in DEFAULT_DEMOGRAPHIC_LABELS]


def prepend_demographic_tokens(
    vocab_entries: list[dict[str, Any]], target_vocab_size: int | None = None
) -> list[dict[str, Any]]:
    """Prepend demographic tokens; cap to target_vocab_size if provided."""
    demos = _default_demographic_vocab_entries()
    existing_labels = {v.get("label") for v in vocab_entries if v.get("type") == "demographic"}
    to_add = [e for e in demos if e["label"] not in existing_labels]
    combined = to_add + vocab_entries
    target_size = target_vocab_size if target_vocab_size is not None else len(vocab_entries)
    if len(combined) > target_size:
        return combined[:target_size]
    return combined


# =============================================================================
# Factorized tokenization vocabulary helpers
# =============================================================================


def _factorized_quantile_vocab_entries(num_quantiles: int = 10) -> list[dict[str, Any]]:
    """Build quantile token entries Q:1...Q:n and Q:UNK.

    These are prepended to vocabulary to ensure stable low IDs.

    Args:
        num_quantiles: Number of quantile buckets

    Returns:
        List of vocab entries with type 'quantile'
    """
    return [{"type": "quantile", "label": lbl, "weight": -1.0} for lbl in quantile_labels(num_quantiles)]


def _factorized_stage_vocab_entries(discovered_stages: list[str]) -> list[dict[str, Any]]:
    """Build stage token entries from discovered stages plus STAGE:UNK.

    Args:
        discovered_stages: Stage values found during training

    Returns:
        List of vocab entries for STAGE:* tokens
    """
    entries = [
        {"type": "stage", "label": f"STAGE:{s.lower()}", "weight": -1.0}
        for s in sorted(set(discovered_stages))
    ]
    # Always include UNK
    entries.append({"type": "stage", "label": STAGE_UNK_LABEL, "weight": -1.0})
    return entries
