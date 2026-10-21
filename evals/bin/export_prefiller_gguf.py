"""Compatibility entry point for the standalone prefiller tools."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
from export_gguf import ProjectionReader, export_gguf, main, unpack_ggml_bytes, verify_block


if __name__ == "__main__":
    main()
