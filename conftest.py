"""Put `src/` on sys.path for the test suite.

The modules import each other flatly (`from game import State`) and every entry
point is invoked from the repository root, so the package layout is kept as a
plain directory on the path rather than rewriting every import into a package.
Data files stay addressed relative to the root, so `python -m pytest` and
`python src/generate.py` both resolve `data/...` the same way.
"""

import sys
from pathlib import Path

SRC = str((Path(__file__).resolve().parent / "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
