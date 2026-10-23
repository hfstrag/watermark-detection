"""Project-wide paths.

The data directory defaults to ``<project>/data`` and can be overridden with the
``FIREFLY_DATA_DIR`` environment variable (e.g. to point at an external drive).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FIREFLY_DATA_DIR", PROJECT_ROOT / "data"))
SYNTHBUSTER_DIR = DATA_DIR / "synthbuster" / "synthbuster"
RESULTS_DIR = PROJECT_ROOT / "results"
