"""Numeric feature construction for the *embedding* input mode.

'Embedding' here is the model input mode (events -> embedding_text_id + NTP label
+ numeric feature vector), NOT ehr_fm.embedding, which builds the text-embedding
table. These pure functions are the *producer* of the numeric feature vector that
scripts/compute_numeric_stats.py feeds and models/input_encoder.py consumes; the
vector layout is a wire format and must not change without updating both.
"""

import math

# Divides log1p(|v|) to put the majority of clinical values in roughly [0, 1].
# log1p(1000) ≈ 6.9 → 6.9/7 ≈ 1.0.  Values between 5 and 10 all work; 7.0 chosen
# because it maps common lab ranges (platelets ~400, WBC ~10k) close to 1.0.
LOG1P_SCALE_CONSTANT = 7.0

# Ref ranges narrower than this are treated as degenerate (bad data) and fall back
# to the raw log1p path.  Most clinical ranges are >> 0.01 wide.
MIN_REF_RANGE_WIDTH = 0.01


def _compute_numeric_features_ref_range_priority(event):
    """Compute 4-dim institution-invariant numeric feature vector.

    Returns [x_primary, is_refrange, is_log1p, value_present] where:
        value absent/non-finite → [0, 0, 0, 0]  (hard gate → identity)
        ref_range path          → [clip((v-lo)/(hi-lo), -2, 3), 1, 0, 1]
        raw log1p path          → [clip(log1p(|v|)/C, 0, 1.5),  0, 1, 1]

    Exactly one of {is_refrange, is_log1p} is 1.0 when value_present=1.
    value_present occupies the last position so the NumericalEncoder hard gate
    (which reads x[..., -1:]) works unchanged.
    """
    v = event.get("numeric_value")
    if v is None or not math.isfinite(v):
        return [0.0, 0.0, 0.0, 0.0]

    ref_low = event.get("ref_low")
    ref_high = event.get("ref_high")

    ref_range_valid = (
        ref_low is not None
        and ref_high is not None
        and math.isfinite(ref_low)
        and math.isfinite(ref_high)
        and (ref_high - ref_low) > MIN_REF_RANGE_WIDTH
    )

    if ref_range_valid:
        raw_pos = (v - ref_low) / (ref_high - ref_low)
        x_primary = max(-2.0, min(3.0, raw_pos))
        return [x_primary, 1.0, 0.0, 1.0]

    x_primary = max(0.0, min(math.log1p(abs(v)) / LOG1P_SCALE_CONSTANT, 1.5))
    return [x_primary, 0.0, 1.0, 1.0]


def _compute_numeric_features(event, code, numeric_stats, quantile_breaks, numeric_override_mode="none"):
    """Compute 5-dim numeric feature vector for an event.

    numeric_override_mode:
        "none"  — standard computation (z-score + quantile from stats, 0/0.5 fallback for missing codes).
        "zero"  — force value_log_zscore=0 and quantile=0.5 for all events; ref_range_* preserved.
    """
    numeric_value = event.get("numeric_value")
    if numeric_value is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    value_present = 1.0

    if numeric_override_mode == "zero":
        value_log_zscore = 0.0
        quantile = 0.5
    else:
        log_val = math.log1p(abs(numeric_value))
        stats = numeric_stats.get(code)
        if stats:
            log_mean = stats.get("log_mean", 0.0)
            log_std = stats.get("log_std", 1.0)
            value_log_zscore = (log_val - log_mean) / log_std
        else:
            value_log_zscore = 0.0

        breaks = quantile_breaks.get(code, [])
        if breaks:
            q = 0
            for b in breaks:
                if numeric_value <= b:
                    break
                q += 1
            quantile = q / max(len(breaks), 1)
        else:
            quantile = 0.5

    ref_low = event.get("ref_low")
    ref_high = event.get("ref_high")

    if ref_low is not None and ref_high is not None and ref_high > ref_low:
        ref_range_available = 1.0
        ref_range_position = (numeric_value - ref_low) / (ref_high - ref_low)
        ref_range_position = max(-2.0, min(3.0, ref_range_position))
    else:
        ref_range_available = 0.0
        ref_range_position = 0.0

    return [value_log_zscore, quantile, ref_range_position, ref_range_available, value_present]
