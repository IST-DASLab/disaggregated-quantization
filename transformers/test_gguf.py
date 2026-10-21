from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gguf import verify_block
from pack_prefill import P_Q, P_BA, pack_layer_trained, write_block_gguf


def test_gguf_preserves_packed_codes_scales_and_bf16(tmp_path):
    rng = np.random.default_rng(3)
    packed = rng.integers(0, 256, (4, 64), dtype=np.uint8)
    scales = rng.integers(1, 127, (4, 8), dtype=np.uint8)
    bf16 = rng.integers(0, 65536, (4, 128), dtype=np.uint16)
    entries = [(P_Q, 4, 128, packed, scales, 7., 3.),
               (P_BA, 4, 128, bf16, None, 0., 0.)]
    path = tmp_path / "block.gguf"
    write_block_gguf(str(path), 0, True, entries)
    verify_block(path, 0, True, entries)
    changed = [(P_Q, 4, 128, packed, scales, 8., 3.), entries[1]]
    with pytest.raises(ValueError, match="Global scale changed"):
        verify_block(path, 0, True, changed)


def test_legacy_separate_projections_keep_individual_input_scales():
    tensors = {}
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for index, name in enumerate(projections):
        prefix = f"model.language_model.layers.0.{name}"
        tensors[prefix + ".weight_packed"] = torch.full((4, 32), index, dtype=torch.uint8)
        tensors[prefix + ".weight_scale"] = torch.ones((4, 4), dtype=torch.float8_e4m3fn)
        tensors[prefix + ".weight_global_scale"] = torch.tensor([2.])
        tensors[prefix + ".input_global_scale"] = torch.tensor([float(index + 1)])
    class Reader:
        def get(self, name):
            return tensors[name]
    with pytest.raises(AssertionError, match="input_global_scale"):
        pack_layer_trained(Reader(), 0, False, {})
    entries = pack_layer_trained(Reader(), 0, False, {}, validate_groups=False)
    assert len(entries) == 7
    assert [entry[-1] for entry in entries] == list(range(1, 8))
