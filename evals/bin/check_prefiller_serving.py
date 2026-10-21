"""Compatibility entry point for the standalone prefiller tools."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
from check_serving import main


if __name__ == "__main__":
    main()
