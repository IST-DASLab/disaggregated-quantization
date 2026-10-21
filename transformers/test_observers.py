import sys
import os
from pathlib import Path

os.environ["USE_HUB_KERNELS"] = "NO"

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
import prefiller
from observers import activation_observers
from quantizers.base import set_recompute_wq
from quantizers.frozen_decode import apply_nvfp4frozendec


def observer_model():
    set_recompute_wq(True)
    model = torch.nn.Module()
    model.gate_proj = torch.nn.Linear(16, 16, bias=False)
    model.up_proj = torch.nn.Linear(16, 16, bias=False)
    apply_nvfp4frozendec(model)
    return model


def test_legacy_observers_require_explicit_reconciliation():
    model = observer_model()
    model.gate_proj.act_amax.fill_(2)
    model.up_proj.act_amax.fill_(3)
    weights = {name: m.export_tensors("prefill") for name, m in model.named_children()}
    with pytest.raises(ValueError, match="Inconsistent fused activation observers"):
        activation_observers(model)
    changes = activation_observers(model, reconcile=True)
    assert changes and model.gate_proj.act_amax == model.up_proj.act_amax == 3
    for name, module in model.named_children():
        after = module.export_tensors("prefill")
        for key in ("weight_packed", "weight_scale", "weight_global_scale"):
            # FP8 CPU equality is unsupported: compare the encoded bytes.
            assert torch.equal(weights[name][key].view(torch.uint8), after[key].view(torch.uint8))


def distributed_observers(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = observer_model()
        model.gate_proj.act_amax.fill_(rank + 2)
        model.up_proj.act_amax.fill_(rank + 2)
        activation_observers(model, synchronize=True, reconcile=True)
        assert model.gate_proj.act_amax == model.up_proj.act_amax == 3
        dcp.save({"model": model.state_dict()}, checkpoint_id=output)
        state = model.state_dict()
        for name in ("gate_proj.act_amax", "up_proj.act_amax"):
            state[name] = torch.zeros_like(state[name])
        dcp.load({"model": state}, checkpoint_id=output)
        assert state["gate_proj.act_amax"] == state["up_proj.act_amax"] == 3
    finally:
        dist.destroy_process_group()


def test_two_rank_dcp_preserves_synchronized_observers(tmp_path):
    mp.spawn(distributed_observers, args=(str(tmp_path / "rdzv"), str(tmp_path / "save")),
             nprocs=2, join=True)


def test_frozen_observers_do_not_follow_training_batches():
    model = observer_model().train()
    from quantizers.dual import quant_phase
    for module in model.children():
        module.act_amax.fill_(4)
        module._observed = True
        module.observer_frozen = True
    with quant_phase(torch.ones(1, 2, dtype=torch.bool)):
        for module in model.children():
            module._quant_act(torch.full((1, 2, 16), 100.))
            assert module.act_amax == 4
            # Calibration remains an explicit operation.
            module.calibrating = True
            module._quant_act(torch.full((1, 2, 16), 100.))
            assert module.act_amax == 100


def test_real_export_from_checkpoint_is_validated_and_atomic(tmp_path):
    import json
    import os
    import subprocess
    from test_prefiller import base_model
    from prefiller import prepare_student
    from safetensors.torch import save_file
    from export_validation import validate_export

    student = base_model()
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    save_file({k: v.detach().clone() for k, v in student.state_dict().items()},
              str(decoder / "model.safetensors"))
    prepare_student(student, decoder, "model.")
    for module in student.modules():
        if hasattr(module, "act_amax"):
            module.act_amax.fill_(4)
    config = tmp_path / "student-config"
    student.config.save_pretrained(config)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    from train import tiny_tokenizer
    tiny_tokenizer().save_pretrained(checkpoint)
    torch.save({"student." + k: v for k, v in student.state_dict().items()},
               checkpoint / "pytorch_model.bin")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    output = tmp_path / "prefill-nvfp4"
    command = [sys.executable, str(Path(__file__).with_name("export_prefiller.py")),
               "--checkpoint", str(checkpoint), "--student-config", str(config / "config.json"),
               "--output-dir", str(output)]
    subprocess.run(command, check=True, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
    report = json.loads((output / "export_validation.json").read_text())
    assert report["packed_nvfp4_linears"] == 13
    assert not list(tmp_path.glob(".nvfp4-export-*"))
    # Validation must reject any accidental reversion to incompatible fusion.
    assert validate_export(output, expected_linears=13)["structural_validation"] == "passed"
