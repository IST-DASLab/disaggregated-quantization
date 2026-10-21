"""Compatibility entry point for the standalone prefiller tools."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
from prepare_multimodal import Shards, main, outer_name


if __name__ == "__main__":
    main()
