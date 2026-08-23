"""Test fixtures. The library under test depends on torch alone; QAD is optional.

tests/test_vs_qad.py cross-checks the packed format against the training code that
produced the weights. That check needs the QAD tree importable, which it is when these
tests run from inside the repo. Installed standalone it is not, and those tests skip --
the rest of the suite still proves the format is internally consistent.
"""

import os
import sys

import pytest
import torch

_QAD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _qad_importable() -> bool:
    if not os.path.isdir(os.path.join(_QAD, "quantizers")):
        return False
    if _QAD not in sys.path:
        sys.path.insert(0, _QAD)
    try:
        import quantizers.blocked  # noqa: F401
        return True
    except Exception:
        return False


HAVE_QAD = _qad_importable()
HAVE_GPU = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(not HAVE_GPU, reason="needs a CUDA device")
requires_qad = pytest.mark.skipif(not HAVE_QAD, reason="QAD tree not importable")


def pytest_report_header(config):
    import lloyd43
    return [f"lloyd43 from: {lloyd43.__file__}",
            f"gpu: {torch.cuda.get_device_name(0) if HAVE_GPU else 'none'}",
            f"qad cross-check: {'on' if HAVE_QAD else 'SKIPPED'}"]


@pytest.fixture
def device():
    return "cuda" if HAVE_GPU else "cpu"


def rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()
