"""Compatibility entry point; importing the standalone module installs the worker patch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
from serve_prefiller import install, projection_groups


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    main()
