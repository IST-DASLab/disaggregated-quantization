"""Export the trained prefill half from a Trainer FSDP checkpoint to packed NVFP4."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE

from train import load_text_model, allocate_empty_parameters
from prefiller import prepare_student
from quantizers.frozen_decode import NVFP4FrozenDecodeLinear
from export.save import save_checkpoint


def restore_student(student, checkpoint):
    """Load only student tensors from DCP; no teacher or decoder model is reloaded."""
    prepare_student(student, None, load_weights=False)
    allocate_empty_parameters(student)
    state = {f"student.{key}": value for key, value in student.state_dict().items()}
    dcp.load({"model": state}, checkpoint_id=str(Path(checkpoint) / "pytorch_model_fsdp_0"))
    student.load_state_dict({key.removeprefix("student."): value for key, value in state.items()})
    for module in student.modules():
        if isinstance(module, NVFP4FrozenDecodeLinear):
            module._observed = bool(module.act_amax > 0)
    student.eval()
    return student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.8-27B-FP8")
    args = parser.parse_args()
    hf_cache = HF_HUB_CACHE
    student = load_text_model(args.teacher_model, True, hf_cache, metadata_only=True)
    restore_student(student, args.checkpoint)
    student.config.dtype = torch.bfloat16
    step = json.loads((Path(args.checkpoint) / "trainer_state.json").read_text())["global_step"]
    save_checkpoint(student, Path(args.output_dir), variant="prefill", step=step)
    AutoTokenizer.from_pretrained(args.teacher_model, cache_dir=hf_cache).save_pretrained(args.output_dir)
    print(f"NVFP4 prefiller exported to {args.output_dir}")


if __name__ == "__main__":
    main()
