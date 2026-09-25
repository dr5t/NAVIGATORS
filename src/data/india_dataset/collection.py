"""Append-only acquisition sink for phone/native sensor adapters.

The adapter supplies observed values and explicit loss annotations. No location,
identity, sampling rate, units, or reference trajectory is invented here.
"""
import json
import os
from pathlib import Path


class SessionCollector:
    """Create raw JSONL plus immutable metadata sidecar using exclusive writes.

    Closing records does not certify a session: build_release validates it.
    A crash leaves the raw prefix available for auditing and validation.
    """

    def __init__(self, path, metadata):
        self.path = Path(path)
        if self.path.suffix != ".jsonl":
            raise ValueError("Collection path must end in .jsonl")
        encoded = json.dumps(metadata, allow_nan=False, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        with Path(str(self.path) + ".metadata.json").open("x") as file:
            file.write(encoded + "\n")
        self._file = self.path.open("x")

    def append(self, sample):
        if not isinstance(sample, dict):
            raise ValueError("Sample must be an object")
        self._file.write(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
