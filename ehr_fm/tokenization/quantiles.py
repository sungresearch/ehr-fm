"""Quantile bucket assignment for factorized tokenization.

Provides functions to assign numeric values to quantile buckets based on
precomputed breaks from vocabulary training.
"""

from .tokens import quantile_token, quantile_unk_token


def assign_quantile_bucket(
    code: str,
    numeric_value: float,
    quantile_breaks: dict[str, list[float]],
) -> str:
    """Assign numeric value to quantile bucket.

    Uses precomputed breaks from vocabulary training.
    Bucket assignment: value >= break[i] means bucket i+2.

    Args:
        code: Event code to look up breaks for
        numeric_value: Value to assign
        quantile_breaks: {code: [break1, break2, ...]} from training

    Returns:
        Token string: Q:<k> or Q:UNK if code not in breaks

    Examples:
        breaks = [70, 85, 95, 105]  # 5 buckets
        value=60  → Q:1 (below first break)
        value=70  → Q:2 (>= first break)
        value=90  → Q:3 (>= second break)
        value=110 → Q:5 (>= all breaks)

        breaks = []  # Invariant values
        value=100 → Q:1 (all values go to bucket 1)
    """
    if code not in quantile_breaks:
        return quantile_unk_token()

    breaks = quantile_breaks[code]

    # Empty breaks = invariant values (all samples identical)
    # All values go to bucket 1
    if not breaks:
        return quantile_token(1)

    bucket = 1
    for brk in breaks:
        if numeric_value >= brk:
            bucket += 1
        else:
            break

    return quantile_token(bucket)
