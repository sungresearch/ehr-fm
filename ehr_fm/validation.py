import errno
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator


class PathValidator(BaseModel, extra="forbid"):
    path: Path
    ptype: Literal["file", "dir"] = None
    extensions: list[str] | str = None

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if isinstance(self.extensions, str):
            self.extensions = [self.extensions]
        if (self.ptype == "file" and not self.path.is_file()) or (
            self.ptype == "dir" and not self.path.is_dir()
        ):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)
        if self.extensions:
            assert any(
                self.path.suffix == ext for ext in self.extensions
            ), f"Invalid path extension '{self.path.suffix}', expected one of '{self.extensions}'."

        return self


class ConfigValidator(BaseModel):
    config: Path | dict | BaseModel
    model: type[BaseModel]

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        from .io import read_json_yaml

        if isinstance(self.config, Path):
            config = read_json_yaml(self.config)
            if not config:
                raise ValueError(f"Configuration file '{self.config}' is empty.")
            self.config = self.model.model_validate(config)

        elif isinstance(self.config, dict):
            self.config = self.model.model_validate(self.config)

        elif not isinstance(self.config, self.model):
            raise TypeError(f"config must be a dict, Path, or instance of {self.model.__name__}.")

        return self


def validate_config(config: str | Path | dict | BaseModel, model: type[BaseModel]) -> BaseModel:
    return ConfigValidator.model_validate(
        {
            "config": config,
            "model": model,
        }
    ).config


class MEDSReaderDatasetConfig(BaseModel):
    meds_reader_path: Path | str
    samples_path: Path | str | None = None
    split: Literal["train", "val", "test", "validation", None] = None
    transform: Callable = None

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if isinstance(self.meds_reader_path, str):
            self.meds_reader_path = Path(self.meds_reader_path)
        if isinstance(self.samples_path, str):
            self.samples_path = Path(self.samples_path)

        PathValidator(path=self.meds_reader_path, ptype="dir")

        if not self.samples_path:
            self.samples_path = self.meds_reader_path / "metadata" / "samples.parquet"
        PathValidator(path=self.samples_path, ptype="file", extensions=".parquet")
        return self


class VocabConfig(BaseModel):
    n_samples: int
    vocab_size: int
    n_numeric_bins: int = 10
    numeric_reservoir_size: int = 10000
    seed: int = None
    numeric_bin_by_unit: bool = False
    min_samples_per_unit: int | None = None
