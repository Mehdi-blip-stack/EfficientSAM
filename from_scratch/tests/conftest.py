"""Make the standalone ``from_scratch`` package importable during pytest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
