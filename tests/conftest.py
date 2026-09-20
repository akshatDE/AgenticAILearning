"""Make the package importable from a source checkout.

A no-op once AgenticGPT is installed with `pip install -e .`, but it means
`pytest` works on a fresh clone without one.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
