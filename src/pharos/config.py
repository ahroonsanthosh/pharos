"""Load frozen dataclass configs from YAML, rejecting unknown keys."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def from_dict(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Build dataclass ``cls`` from a (possibly nested) mapping."""
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        kind = hints[key]
        if dataclasses.is_dataclass(kind) and isinstance(value, Mapping):
            value = from_dict(kind, value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_yaml(cls: type[T], path: str | Path) -> T:
    return from_dict(cls, yaml.safe_load(Path(path).read_text()))
