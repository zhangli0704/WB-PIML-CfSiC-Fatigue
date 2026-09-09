"""Compatibility launcher for the unchanged 0909 analysis source."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if not any(arg == "--out" or arg.startswith("--out=") for arg in sys.argv[1:]):
        sys.argv.extend(["--out", str(root / "results")])
    runpy.run_path(str(root / "WB-PIML.py"), run_name="__main__")
