"""Numeric feature construction for the *embedding* input mode.

'Embedding' here is the model input mode (events -> embedding_text_id + NTP label
+ numeric feature vector), NOT ehr_fm.embedding, which builds the text-embedding
table. This pure function is the *producer* of the 4-dim institution-invariant
numeric feature vector that the FiLM numeric pathway (models/input_encoder.py)
consumes; the vector layout is a wire format and must not change without updating
both producer and consumer.
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
