"""Ticker news collection and an optional, separate headline experiment workspace."""

import hashlib
from pathlib import Path

__version__ = "0.5.0"

_CODE_ROOT = Path(__file__).parent
RUNTIME_CODE_SHA256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _CODE_ROOT.glob("*.py")}


def assert_runtime_current():
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _CODE_ROOT.glob("*.py")}
    if current != RUNTIME_CODE_SHA256:
        raise ValueError("Collector code changed while this process was running. Restart the dashboard and collection worker before collecting or exporting; saved evidence is preserved.")
