"""Export the trained prefill half from a Trainer FSDP checkpoint to packed NVFP4."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from train import load_text_model, allocate_empty_parameters
from prefiller import prepare_student, TextOnly
from quantizers.frozen_decode import NVFP4FrozenDecodeLinear
from export.save import save_checkpoint


def restore_student(student, checkpoint):
    """Load only student tensors from DCP; no teacher or decoder model is reloaded."""
    prepare_student(student, None, load_weights=False)
    allocate_empty_parameters(student)
    state = {f"student.{key}": value for key, value in student.state_dict().items()}
    sharded = Path(checkpoint) / "pytorch_model_fsdp_0"
    if sharded.is_dir():
        dcp.load({"model": state}, checkpoint_id=str(sharded))
    else:
        loaded = torch.load(Path(checkpoint) / "pytorch_model.bin", mmap=True, weights_only=True)
        state = {key: loaded[key] for key in state}
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
    parser.add_argument("--student-config", type=Path,
                        help="Saved text config; avoids rebuilding the teacher/vision model")
    parser.add_argument("--multimodal", action="store_true",
                        help="Attach the teacher's frozen BF16 vision tower and processor")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Replace a validated export from the same run after successful packing")
    parser.add_argument("--reconcile-activation-observers", action="store_true",
                        help="Explicitly use surviving fused-group maxima from legacy FSDP saves; "
                             "this changes activation calibration, not packed weights")
    args = parser.parse_args()
    if args.student_config is None:
        saved_config = Path(args.checkpoint).parent / "student-config/config.json"
        if saved_config.is_file():
            args.student_config = saved_config
    hf_cache = HF_HUB_CACHE
    output = Path(args.output_dir).resolve()
    if output.exists():
        provenance = output / "activation_observers.json"
        if not (args.replace_existing and provenance.is_file() and
                (output / "export_validation.json").is_file()):
            raise ValueError(f"Refusing to replace existing export {output}")
        source = Path(json.loads(provenance.read_text())["checkpoint"])
        if source.parent.resolve() != Path(args.checkpoint).resolve().parent:
            raise ValueError("Existing export belongs to a different training run")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.student_config:
        from accelerate import init_empty_weights
        config = AutoConfig.from_pretrained(args.student_config.parent)
        with init_empty_weights(include_buffers=False):
            student = TextOnly(AutoModelForCausalLM.from_config(config, dtype=torch.float32,
                                                              attn_implementation="sdpa"))
    else:
        student = load_text_model(args.teacher_model, True, hf_cache, metadata_only=True)
    restore_student(student, args.checkpoint)
    from observers import activation_observers
    changes = activation_observers(student, reconcile=args.reconcile_activation_observers)
    student.config.dtype = torch.bfloat16
    student.config.use_cache = True
    step = json.loads((Path(args.checkpoint) / "trainer_state.json").read_text())["global_step"]
    # Publish only after packing, assets and validation succeed. A failed export
    # leaves the resumable checkpoint intact and no half-written serving path.
    with tempfile.TemporaryDirectory(prefix=".nvfp4-export-", dir=output.parent) as temporary:
        text = Path(temporary) / "text"
        save_checkpoint(student, text, variant="prefill", step=step)
        packed_count = sum(isinstance(m, NVFP4FrozenDecodeLinear) for m in student.modules())
        if (Path(args.checkpoint) / "tokenizer_config.json").is_file():
            AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True).save_pretrained(text)
        elif not args.student_config:
            AutoTokenizer.from_pretrained(args.teacher_model, cache_dir=hf_cache).save_pretrained(text)
        ready = text
        if args.multimodal:
            base = snapshot_download(args.teacher_model, cache_dir=hf_cache, local_files_only=True)
            ready = Path(temporary) / "servable"
            root = Path(__file__).resolve().parent
            subprocess.run([sys.executable, str(root / "prepare_multimodal.py"),
                            "--base", base, "--text-export", str(text), "--output", str(ready)], check=True)
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(args.teacher_model, cache_dir=hf_cache).save_pretrained(ready)
        (ready / "activation_observers.json").write_text(json.dumps({
            "checkpoint": str(Path(args.checkpoint).resolve()), "reconciled": changes,
            "limitation": "Unsaved rank-local observers cannot be recovered from legacy checkpoints."
                          if changes else None,
        }, indent=2))
        from export_validation import validate_export
        report = validate_export(ready, expected_linears=packed_count)
        report["serving_entrypoint"] = ("vllm serve" if getattr(student.config, "prefiller_fuse_gdn_scales", False)
                                         else "python serve_prefiller.py serve")
        report["shared_decoder_norms"] = getattr(student.config, "prefiller_shared_decoder_norms", False)
        report["llama_cpp_norm_contract_compatible"] = report["shared_decoder_norms"]
        report["handoff_parity"] = "Requires runtime verification; NIXL and linked ODP use different token boundaries."
        report["handoff_boundary"] = getattr(student.config, "prefiller_handoff_boundary", "prompt-end")
        (ready / "export_validation.json").write_text(json.dumps(report, indent=2))
        previous = Path(temporary) / "previous"
        if output.exists():
            os.rename(output, previous)
        try:
            os.rename(ready, output)
        except BaseException:
            if previous.exists():
                os.rename(previous, output)
            raise
    print(f"Real packed NVFP4 prefiller exported and validated: {output}")


if __name__ == "__main__":
    main()
