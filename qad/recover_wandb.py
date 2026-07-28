"""Rebuild the wandb runs for the three gsqlloyd3bit logit_lr arms from job logs.

Those jobs ran before WANDB_DIR was set, so wandb had no writable run directory and
never persisted any metrics (the one offline-run dir on disk is empty). The curves
survive only in the SLURM logs: the tqdm postfix carries per-step kl/lr/ntp/delta,
and stdout carries val_ntp every 25 steps. New run ids are used because the ids the
jobs printed belong to runs that have since been deleted.
"""
import glob, json, os, re, sys
import wandb

LOGDIR = "../logs/train"
ARMS = {  # job id -> (quant hash, logit_lr)
    "474412": ("34a2e5a5", 3e-5),
    "474413": ("113e0057", 1e-4),
    "474414": ("28a6f8d3", 3e-4),
}
TRAIN_RE = re.compile(
    r"(\d+)/2485 .*?kl=([0-9.]+), lr=([0-9.eE+-]+), ntp=([0-9.]+)(?:, Δ=([+-][0-9.]+))?")
VAL_RE = re.compile(r"step +(\d+) \| val_ntp=([0-9.]+) +teacher=([0-9.]+)")

for job, (qhash, llr) in ARMS.items():
    errs = glob.glob(f"{LOGDIR}/*/*_{job}.err")
    if not errs:
        print(f"{job}: no log found"); continue
    err = errs[0]; out = err[:-4] + ".out"

    train = {}
    with open(err, errors="ignore") as fh:
        for line in fh.read().replace("\r", "\n").split("\n"):
            m = TRAIN_RE.search(line)
            if m:
                s = int(m.group(1))
                train[s] = dict(kl=float(m.group(2)), lr=float(m.group(3)),
                                ntp=float(m.group(4)),
                                delta=float(m.group(5)) if m.group(5) else None)
    val = {}
    with open(out, errors="ignore") as fh:
        for m in VAL_RE.finditer(fh.read()):
            val[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

    tag = f"qad3x-Qwen-Qwen3-0.6B-gsqlloyd3bit-{qhash}"
    print(f"{job} {tag}: {len(train)} train steps, {len(val)} val points -> uploading")
    run = wandb.init(
        project="prefill-decode-distill",
        id=f"{tag}-recovered",              # fresh id; the original was deleted
        name=f"{tag} (recovered)",
        resume="never",
        config=dict(model="Qwen/Qwen3-0.6B", quantizer="gsqlloyd3bit",
                    quantizer_hash=qhash, run_prefix="qad3x",
                    train_tokens=100_000_000, lr=3e-6, lr_schedule="constant",
                    warmup_steps=100, global_batch_size=64,
                    quantizer_params=dict(logit_lr=llr, scale_lr=3e-6, optim="lion",
                                          grid="lloyd", block_size=16,
                                          logits_dtype="fp32"),
                    recovered_from_logs=True, source_job=job),
    )
    for s in sorted(train):
        p = {f"train/{k}": v for k, v in train[s].items() if v is not None}
        p["train/kl"] = train[s]["kl"]
        if s in val:
            p["val/ntp_loss"] = val[s][0]
            p["val/teacher_ntp"] = val[s][1]
            p["val/ntp_delta"] = val[s][0] - val[s][1]
        wandb.log(p, step=s)
    wandb.summary["final/val_ntp"] = val[max(val)][0] if val else None
    wandb.finish()
    print(f"  done: {run.url}")
