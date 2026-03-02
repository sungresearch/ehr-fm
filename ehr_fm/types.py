from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

PathLike = str | Path
ConfigLike = PathLike | dict | BaseModel
EventDict = dict[str, Any]  # {"code": str, "numeric_value": float | None, ...}
EventSequence = Sequence[EventDict]
