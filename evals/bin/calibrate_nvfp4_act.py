"""Calibrate NVFP4 activation scales onto an already-NVFP4 ModelOpt checkpoint.

    python3 calibrate_nvfp4_act.py --src <NVFP4 dir> --dst <out dir> [--nsamples 256]

Turns a ModelOpt W4A16_NVFP4 checkpoint into a W4A4 NVFP4 one by measuring the
activation amax feeding every quantized linear and writing the per-module
`input_scale` that vLLM requires. Weights are NOT touched.

WHY THIS CANNOT BE A CONFIG EDIT
--------------------------------
kimi-k3's W4A4 arm is a pure config edit because MXFP4 activations are dynamic
(observer null): the block scales are computed per token at runtime and nothing needs
calibrating. NVFP4 activations are static, and vLLM's ModelOptNvFp4LinearMethod says so
outright:

    if not self.quant_config.is_checkpoint_nvfp4_serialized:
        raise ValueError("NVFP4 quantization was selected, "
                         " dynamic quantization is not supported.")

It then registers `input_scale` as a required fp32 PerTensorScaleParameter. There is no
dynamic fallback, so the 5,935 values have to be measured.

THE SCALE CONVENTION, WHICH IS THE WHOLE BALLGAME
-------------------------------------------------
    input_scale = amax_activation / (6 * 448)

That is E2M1_MAX * E4M3_MAX, and it comes from ModelOpt's own exporter
(nvfp4_tensor.py::get_activation_scaling_factor):

    activation_scaling_factor = amax.float() / (quantizer.maxbound * E4M3_MAX)

DO NOT copy the divisor from this checkpoint's weight_scale_2. Its weights were
quantized in ModelOpt's `four_over_six` mode, where the weight global scale uses 256
rather than 448:

    def fp8_max_for_normalization(quantizer) -> float:
        \"\"\"FP8 normalization max: 256 for 4/6, else 448.\"\"\"
        return E4M3_MAX_46 if bs.get("four_over_six", False) else E4M3_MAX

Measured on this checkpoint, stored weight_scale_2 is exactly 1.75x = 448/256 above
amax/(6*448) -- for layers.1.mixer.experts.0.down_proj, amax/(6*256) = 1.38601e-04
against a stored 1.38601e-04. The activation path has NO such branch; E4M3_MAX is
hard-coded there. Using 256 for activations would leave the scale 1.75x too large and
waste most of the FP4 range on every tensor.

WHAT GETS CALIBRATED
--------------------
The 5,935 modules ModelOpt tagged W4A16_NVFP4: 2,944 expert down_proj + 2,944 expert
up_proj (23 MoE layers x 128 experts), 23 shared-expert pairs, and lm_head. The 46
mamba in_proj/out_proj are already FP8 W8A8 with their own input_scale and are left
alone; attention q/k/v/o and the routers are in `ignore` and stay bf16.

lm_head is EXCLUDED by default (--include-lm-head to override). It is in ModelOpt's
W4A16 set, but 4-bit activations on the final hidden state perturb every logit at once,
and NVIDIA's own W4A4 build for the sibling omni model quantizes the routed experts
only. Excluding it keeps it W4A16, which is what it already is -- this is a smaller
change, not an extra one.

WEIGHTS ARE SYMLINKED, NOT COPIED
---------------------------------
The output is the original safetensors symlinked, plus ONE new shard holding the
input_scale tensors, plus a rewritten index and config. 24 KB of new data against 21 GB
of weights, and the weights are provably identical because they are the same inodes.

A DOCUMENTED RISK, STATED UP FRONT
----------------------------------
Every W4A4 checkpoint this repo's calibration produced for nemotron-3-omni-30b was
degenerate -- 16/16 smoke items empty at the token cap, on three different module sets,
while NVIDIA's own NVFP4 build of the same architecture family worked. The recorded
suspect was this repo's calibration. Lightning is the same nemotron_h family. Smoke test
before spending anything: --validate generates from the dequantized model so a broken
dequant is caught here rather than after a 4 h eval returns empty strings.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

E2M1_MAX = 6.0
E4M3_MAX = 448.0

# FP4 E2M1 code -> value. Index is the 4-bit code; the top bit is sign.
FP4_LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)


def dequantize_nvfp4(packed, block_scale, global_scale, group_size=16):
    """Packed uint8 NVFP4 -> bf16. Two 4-bit codes per byte, low nibble first."""
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    lut = FP4_LUT.to(packed.device)
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(packed.shape[0], -1)
    scales = block_scale.to(torch.float32).repeat_interleave(group_size, dim=1)
    return (vals * scales * global_scale.float().reshape(-1, 1)).to(torch.bfloat16)


def quantized_layers(cfg_dict):
    qc = cfg_dict["quantization_config"]
    return qc["quantized_layers"], qc


def build_state_dict(src, targets):
    """Every tensor, NVFP4 ones reconstructed to bf16, experts stacked, mtp dropped.

    TWO CHECKPOINT-vs-GRAPH MISMATCHES, both silent under strict=False:

    1. transformers 5.16 stores all 128 experts of a layer as ONE 3D parameter --
       NemotronHExperts holds up_proj (num_experts, intermediate, hidden) and calls
       F.linear(x, self.up_proj[i]) in a loop. The checkpoint stores 2,944 separate 2D
       tensors named experts.<i>.up_proj. They have to be stacked in expert order.

    2. The checkpoint carries an mtp.* multi-token-prediction module (6,158 tensors) that
       AutoModelForCausalLM does not build at all. It is already in the quant config's
       `ignore` list; dropping it here is the same decision applied to loading.
    """
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(idx.values()))
    sd, experts = {}, {}
    EXP = re.compile(r"^(.*\.mixer\.experts)\.(\d+)\.(up_proj|down_proj)\.weight$")
    for i, shard in enumerate(shards):
        t = load_file(os.path.join(src, shard))
        mods = {k.rsplit(".", 1)[0] for k in t if k.endswith(("weight_scale", "weight_scale_2"))}
        for m in sorted(mods):
            w, bs = t.get(f"{m}.weight"), t.get(f"{m}.weight_scale")
            gs = t.get(f"{m}.weight_scale_2")
            if w is None or bs is None:
                continue
            if w.dtype == torch.uint8 and gs is not None and m in targets:
                group = w.shape[1] * 2 // bs.shape[1]
                t[f"{m}.weight"] = dequantize_nvfp4(w, bs, gs, group)
            elif w.dtype == torch.float8_e4m3fn:
                # The 46 mamba in_proj/out_proj are FP8 W8A8 -- a per-tensor
                # weight_scale, NOT raw fp8 values. Casting fp8 -> bf16 and dropping the
                # scale silently rescales 23 of 52 layers by a per-tensor factor, which
                # is what produced
                #   VALIDATE: 'The capital of France is northeter school! german ...'
                # after the prefix remap was already correct. These layers are not being
                # calibrated, but they still have to be RIGHT for the activations
                # downstream of them to mean anything.
                t[f"{m}.weight"] = (w.to(torch.float32)
                                    * bs.to(torch.float32).reshape(-1, 1)).to(torch.bfloat16)
            else:
                continue
            for suf in ("weight_scale", "weight_scale_2"):
                t.pop(f"{m}.{suf}", None)
        for k, v in t.items():
            if k.startswith("mtp."):
                continue
            if k.endswith(("weight_scale", "weight_scale_2", "input_scale", "k_scale", "v_scale")):
                continue
            if v.dtype == torch.float8_e4m3fn:
                v = v.to(torch.bfloat16)
            m = EXP.match(k)
            if m:
                experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = v
            else:
                sd[k] = v
        print(f"  shard {i + 1}/{len(shards)}", flush=True)
    for (prefix, proj), byidx in experts.items():
        n = max(byidx) + 1
        if len(byidx) != n:
            raise SystemExit(f"{prefix}.{proj}: {len(byidx)} experts for indices 0..{n-1}")
        sd[f"{prefix}.{proj}"] = torch.stack([byidx[i] for i in range(n)])
    print(f"  stacked {len(experts)} fused expert tensors")
    return sd


def calibration_texts(tokenizer, nsamples, seqlen):
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    pat = os.path.join(hub, "datasets--HuggingFaceH4--ultrachat_200k", "snapshots",
                       "*", "data", "*train_sft*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no ultrachat parquet under {pat}")
    import pyarrow.parquet as pq
    rows, out = [], []
    for f in files:
        rows.extend(pq.read_table(f, columns=["messages"]).to_pylist())
        if len(rows) >= nsamples * 2:
            break
    for r in rows:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            continue
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] < 64:
            continue
        out.append(ids[:, :seqlen])
        if len(out) >= nsamples:
            break
    if len(out) < nsamples:
        print(f"WARNING: only {len(out)} usable samples (asked {nsamples})")
    return out


def write_out(args, cfg_dict, scales, hooked):
    # --- write ------------------------------------------------------------
    os.makedirs(args.dst, exist_ok=True)

    # Scales FIRST. Everything above this point is ~14 minutes of GPU work and the copy
    # loop below is the cheap part; writing the expensive artefact last means a trivial
    # error down here throws the measurement away. It already did once, on a stray
    # .eval_results DIRECTORY that shutil.copy2 cannot handle.
    extra = "model-input-scales.safetensors"
    save_file(scales, os.path.join(args.dst, extra))
    print(f"  wrote {extra}")

    for f in os.listdir(args.src):
        sp, d = os.path.join(args.src, f), os.path.join(args.dst, f)
        if os.path.exists(d) or f.startswith("."):
            continue
        if f.endswith(".safetensors"):
            os.symlink(os.path.realpath(sp), d)
        elif os.path.isdir(sp):
            continue
        elif f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(sp, d)

    idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    idx["weight_map"].update({k: extra for k in scales})
    json.dump(idx, open(os.path.join(args.dst, "model.safetensors.index.json"), "w"),
              indent=2)

    for n in hooked:
        cfg_dict["quantization_config"]["quantized_layers"][n]["quant_algo"] = "NVFP4"
    json.dump(cfg_dict, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

    print(f"\nwrote {args.dst}")
    print(f"  {len(hooked)} modules relabelled W4A16_NVFP4 -> NVFP4")
    print(f"  {len(scales)} input_scale tensors in {extra}")
    print("  weights symlinked, byte-identical to the source")
    print("\nConfirm the arm took effect from the server log: the NvFp4 MoE backend "
          "must read FLASHINFER_TRTLLM (W4A4), not MARLIN (weight-only).")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--validate", action="store_true")
    # The VENDOR recipe does not calibrate these at all. ModelOpt's config pins the
    # expert input quantizer to a constant:
    #     - quantizer_name: '*mixer.experts*input_quantizer'
    #       enable: true
    #       cfg: {$import: nvfp4, constant_amax: 2688.0}
    # and 2688 is exactly E2M1_MAX * E4M3_MAX, so input_scale = 2688/2688 = 1.0 exactly.
    # A global scale of 1 turns NVFP4 activations into pure per-block scaling: the fp8
    # block scale is block_amax/6 with no global rescale, which tolerates activations up
    # to 2688 instead of clipping at whatever amax a calibration set happened to observe.
    # No forward passes, no data, no GPU -- so it is also the honest baseline to measure
    # a measured-amax calibration against.
    ap.add_argument("--constant-amax", type=float, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dict = json.load(open(os.path.join(args.src, "config.json")))
    qlayers, qc = quantized_layers(cfg_dict)

    # Accept a source that is ALREADY relabelled to NVFP4. A checkpoint can declare
    # W4A4 while carrying no input_scale -- that is exactly the state that makes vLLM
    # run on torch.empty and emit nothing -- so those modules are legitimate targets for
    # writing the scales, not an error.
    ok = {"W4A16_NVFP4", "NVFP4"}
    targets = {m for m, v in qlayers.items() if v.get("quant_algo") in ok}
    skipped_lm_head = False
    if not args.include_lm_head and "lm_head" in targets:
        targets.discard("lm_head")
        skipped_lm_head = True
    print(f"targets: {len(targets)} modules (lm_head "
          f"{'excluded' if skipped_lm_head else 'included'})")

    if args.constant_amax is not None:
        a = args.constant_amax / (E2M1_MAX * E4M3_MAX)
        scales = {f"{n}.input_scale": torch.tensor([a], dtype=torch.float32)
                  for n in sorted(targets)}
        print(f"constant amax {args.constant_amax} -> input_scale {a} "
              f"for {len(scales)} modules (no calibration run)")
        write_out(args, cfg_dict, scales, sorted(targets))
        return

    print("loading + dequantizing weights ...", flush=True)
    sd = build_state_dict(args.src, targets | ({"lm_head"} if skipped_lm_head else set()))

    cfg = AutoConfig.from_pretrained(args.src)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

    # PREFIX REMAP, and it is not cosmetic. The checkpoint names everything
    # `backbone.*`; this transformers version builds the graph as `model.*`. With
    # strict=False that mismatch is SILENT -- load_state_dict matches nothing, every
    # weight keeps its random init, and the script happily calibrates noise. It got as
    # far as generating "The capital of France isracaGPtant catgetto Missing'" before
    # anything looked wrong, and only because --validate was there to print it.
    expected = set(model.state_dict())
    # Majority vote on the leading path segment, NOT "do the key sets overlap at all".
    # lm_head.weight is spelled identically under both prefixes, so a plain intersection
    # test is non-empty even when 6,512 of 6,513 keys are misnamed, and the remap is
    # skipped exactly when it is needed.
    unexp = set(sd) - expected
    miss = expected - set(sd)
    remapped, src_pref, dst_pref = False, "", ""
    if unexp and miss:
        import collections
        src_pref = collections.Counter(k.split(".")[0] for k in unexp).most_common(1)[0][0]
        dst_pref = collections.Counter(k.split(".")[0] for k in miss).most_common(1)[0][0]
        if src_pref != dst_pref:
            print(f"remapping checkpoint prefix {src_pref!r} -> {dst_pref!r} "
                  f"({len(unexp)} keys)")
            def remap(k):
                return (dst_pref + k[len(src_pref):]) if k.startswith(src_pref + ".") else k
            sd = {remap(k): v for k, v in sd.items()}
            targets = {remap(t) for t in targets}
            remapped = True

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    real_missing = [k for k in missing if not k.endswith(".weight_scale")]
    # FATAL, not a warning. A partially loaded model produces a checkpoint that looks
    # fine and is calibrated against garbage.
    if real_missing or unexpected:
        print(f"missing={len(real_missing)} e.g. {real_missing[:5]}")
        print(f"unexpected={len(unexpected)} e.g. {unexpected[:5]}")
        raise SystemExit("ERROR: state dict does not match the model graph; refusing to "
                         "calibrate a partially loaded model.")
    del sd
    model.eval().to(dev)
    print(f"model on {dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.src)

    if args.validate:
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False)
        print("VALIDATE:", repr(tok.decode(out[0], skip_special_tokens=True)))

    # --- measurement -------------------------------------------------------
    # NemotronHExperts has NO per-expert submodules: forward loops over
    # F.linear(x, self.up_proj[i]). There is nothing to attach a hook to, and a hook on
    # the block would see only its input -- giving up_proj's activation but never
    # down_proj's, which is act_fn(up_proj(x)) and lives entirely inside the loop.
    #
    # So the loop is reimplemented here, per instance, recording amax for BOTH
    # projections of EVERY expert. That keeps the scales genuinely per-expert, which is
    # what the checkpoint format stores, rather than one scale broadcast over 128
    # experts. Numerically it is the same computation as the original.
    amax = {}

    def record(name, x):
        m = x.detach().abs().amax().float()
        prev = amax.get(name)
        amax[name] = m if prev is None else torch.maximum(prev, m)

    def instrumented(mod, hidden_states, top_k_index, top_k_weights, _pfx=None):
        final = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            em = torch.nn.functional.one_hot(top_k_index, num_classes=mod.num_experts)
            em = em.permute(2, 1, 0)
            hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        for e in hit:
            e = e.item()
            pos, tok = torch.where(em[e])
            if tok.numel() == 0:
                continue
            cur = hidden_states[tok]
            record(f"{_pfx}.{e}.up_proj", cur)
            h = torch.nn.functional.linear(cur, mod.up_proj[e])
            h = mod.act_fn(h)
            record(f"{_pfx}.{e}.down_proj", h)
            h = torch.nn.functional.linear(h, mod.down_proj[e])
            h = h * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
        # back to the input dtype: top_k_weights is fp32, and the real forward is wrapped
        # by @use_experts_implementation which restores it. Replacing forward bypasses
        # that wrapper, so returning fp32 propagated to lm_head and raised
        #   RuntimeError: expected mat1 and mat2 to have the same dtype, float != BFloat16
        return final.to(hidden_states.dtype)

    import functools
    import types
    by_name = dict(model.named_modules())

    # graph name -> original checkpoint name, so scales are keyed the way the index is
    def orig(n):
        return (src_pref + n[len(dst_pref):]) if (remapped and n.startswith(dst_pref + ".")) else n

    n_fused = 0
    for gname, mod in list(by_name.items()):
        if type(mod).__name__ != "NemotronHExperts":
            continue
        mod.forward = types.MethodType(
            functools.partial(instrumented, _pfx=orig(gname)), mod)
        n_fused += 1

    handles, hooked_linear = [], []
    for t in sorted(targets):
        mod = by_name.get(t)
        if mod is None:
            continue
        hooked_linear.append(t)

        def mk(name):
            def hook(_m, inputs):
                record(name, inputs[0])
            return hook
        handles.append(mod.register_forward_pre_hook(mk(orig(t))))

    expected_experts = {t for t in targets if ".experts." in t}
    covered = len(hooked_linear) + len(expected_experts)
    if covered != len(targets):
        raise SystemExit(f"ERROR: {len(targets) - covered} target modules are neither a "
                         f"graph module nor inside a fused expert block.")
    print(f"measuring {len(hooked_linear)} linears + {n_fused} fused expert blocks "
          f"({len(expected_experts)} expert projections)")

    batches = calibration_texts(tok, args.nsamples, args.seqlen)
    print(f"calibrating on {len(batches)} sequences ...", flush=True)
    with torch.no_grad():
        for i, ids in enumerate(batches):
            model(ids.to(dev))
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{len(batches)}  covered={len(amax)}/{len(targets)}",
                      flush=True)
    for h in handles:
        h.remove()
    hooked = [orig(t) for t in targets]

    # --- scales -----------------------------------------------------------
    # An expert that never fired has no amax. Falling back to its layer's median keeps
    # the module servable, but a scale from no data is a guess -- counted and printed
    # so it can never be silently large.
    # Grouped by (layer, PROJECTION), not by layer alone: up_proj sees the residual
    # stream and down_proj sees act_fn(up_proj(x)), which are different distributions.
    # A median pooled across both would hand a down_proj an up_proj-sized scale.
    scales, fallback = {}, []
    group_of = lambda n: (n.split(".mixer.")[0], n.rsplit(".", 1)[-1])
    per_layer = {}
    for n, v in amax.items():
        per_layer.setdefault(group_of(n), []).append(v.item())
    for n in hooked:
        if n in amax:
            a = amax[n].item()
        else:
            vals = sorted(per_layer.get(group_of(n), []))
            if not vals:
                raise SystemExit(f"no calibration data anywhere for layer of {n}")
            a = vals[len(vals) // 2]
            fallback.append(n)
        # SHAPE [1], NOT []. This checkpoint uses both conventions and they are not
        # interchangeable: NVIDIA's own 46 FP8 input_scale tensors (and every k_scale /
        # v_scale) are F32 shape [1], while weight_scale_2 is a true scalar F32 shape [].
        # vLLM loads input_scale through PerTensorScaleParameter, which is indexed per
        # shard, so emitting a rank-0 tensor here would be writing an activation scale in
        # the WEIGHT convention and relying on broadcast to paper over it.
        scales[f"{n}.input_scale"] = torch.tensor([a / (E2M1_MAX * E4M3_MAX)],
                                                  dtype=torch.float32)
    print(f"scales: {len(scales)}  uncovered(fallback)={len(fallback)}")
    if fallback:
        print(f"  e.g. {fallback[:5]}")
    vals = sorted(float(v.reshape(-1)[0]) for v in scales.values())
    print(f"  input_scale min={vals[0]:.4g} p50={vals[len(vals)//2]:.4g} max={vals[-1]:.4g}")

    write_out(args, cfg_dict, scales, hooked)


if __name__ == "__main__":
    sys.exit(main())
