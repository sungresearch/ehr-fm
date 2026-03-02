import json

import yaml

from .logger import setup_logging
from .types import PathLike
from .validation import PathValidator


def read_json_yaml(path: PathLike) -> dict:
    path = PathValidator(path=path, ptype="file", extensions=[".yaml", ".yml", ".json"]).path
    with open(path) as f:
        extension = path.suffix
        if extension in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif extension == ".json":
            return json.load(f)


def write_dict_to_json(data: dict, path: PathLike, overwrite: bool = False) -> None:
    logger = setup_logging()
    path = PathValidator(path=path).path

    if path.is_file() and not overwrite:
        logger.warning(f"Overwrite is set to False. File not saved because it already exists: '{path}'")
        return

    try:
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    except (TypeError, ValueError) as e:
        if path.exists():
            path.unlink()
            logger.warning(f"Failed to serialize data to JSON. Incomplete file '{path}' deleted.")
        logger.error(f"Serialization error: {e}")
        raise

    return
