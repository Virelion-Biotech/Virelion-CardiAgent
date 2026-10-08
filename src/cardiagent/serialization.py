"""Strict, portable JSON and atomic artifact writes."""

import json
import math
import os
import tempfile
from pathlib import Path


def finite_number(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


def positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def strict_json(payload, **kwargs):
    return json.dumps(payload, allow_nan=False, **kwargs)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON number: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=invalid,
        object_pairs_hook=unique_pairs,
    )


def write_json(payload, path):
    text = strict_json(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cardiagent-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
