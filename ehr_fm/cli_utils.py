"""Shared CLI parsing utilities for ehr_fm scripts."""

from __future__ import annotations


def str_to_bool(value: str | bool) -> bool:
    """Convert a string CLI argument to a boolean.

    Accepts Python bools (passthrough for values injected via
    ``parser.set_defaults`` from a YAML/JSON config) and common
    string representations.

    Args:
        value: A boolean or one of the accepted string literals
            (case-insensitive): ``"true"``/``"1"``/``"yes"`` for True,
            ``"false"``/``"0"``/``"no"`` for False.

    Returns:
        The corresponding Python ``bool``.

    Raises:
        argparse.ArgumentTypeError: If *value* is not a recognised boolean string.
    """
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False

    import argparse

    raise argparse.ArgumentTypeError(
        f"Boolean value expected, got '{value}'. "
        "Accepted values: true/false, yes/no, 1/0."
    )


def normalize_resume_checkpoint(value: str | bool | None) -> str | bool | None:
    """Normalise ``resume_from_checkpoint`` from mixed CLI / config sources.

    The argument may arrive as:

    * ``None`` -- no resume (from argparse default or YAML ``null``).
    * Python ``True`` / ``False`` -- from YAML boolean via ``set_defaults``.
    * A truthy/falsy *string* -- from ``--resume_from_checkpoint true`` on CLI
      (argparse ``type=str`` converts everything to a string).
    * An arbitrary string -- treated as a literal checkpoint path.

    Returns:
        ``None`` if the caller should start from scratch, ``True`` if the
        caller should auto-detect a checkpoint, or a ``str`` path to resume
        from a specific checkpoint.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value if value else None

    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no", "none", "null", ""):
        return None
    return value
