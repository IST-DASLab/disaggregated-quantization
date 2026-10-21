"""The source tree must work without an adjacent parent repository."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_project_imports_resolve_inside_this_directory():
    import prefiller
    import train
    import export.save
    import export.config_fix
    import quantizers.frozen_decode
    import quantizers.nvfp4
    import training.data
    import training.models
    import training.pipeline

    for module in (prefiller, train, export.save, export.config_fix,
                   quantizers.frozen_decode, quantizers.nvfp4,
                   training.data, training.models, training.pipeline):
        assert Path(module.__file__).resolve().is_relative_to(ROOT), module.__file__


@pytest.mark.parametrize("script", ["train.py", "launch.py", "export_prefiller.py",
                                    "prepare_multimodal.py", "export_gguf.py", "check_serving.py"])
def test_cli_imports_from_unrelated_working_directory(script, tmp_path):
    # Isolated mode excludes the caller's PYTHONPATH. Add only this source tree
    # as a script invocation would, then preserve normal --help behavior.
    code = ("import runpy,sys; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            f"sys.argv=[{script!r}, '--help']; "
            f"runpy.run_path({str(ROOT / script)!r}, run_name='__main__')")
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_defaults_stay_with_the_source_tree(monkeypatch):
    import train
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-verify-serving"])
    args = train.parse_args()
    assert Path(args.cache_dir).is_relative_to(ROOT)
    assert Path(args.output_dir).is_relative_to(ROOT)
    assert Path(args.serving_python) == ROOT / ".venv-serving/bin/python"
