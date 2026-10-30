"""Test configuration: import the checkout without requiring an install.

``src/`` is the single canonical package location of the release.  Adding it to
``sys.path`` keeps ``pytest`` runnable from a fresh clone before
``pip install -e .`` is done (the same behaviour is declared in
``pyproject.toml`` via ``[tool.pytest.ini_options] pythonpath``).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
