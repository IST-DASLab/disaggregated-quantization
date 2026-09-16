# Migrating this repo to a new cluster

Written 2026-09-09, after losing access to `aws-cmh-slurm-1` (the GB300/aarch64 cluster
this was running on). Consolidates what the 2026-09-01 b200→b300 move already found plus
everything the RULER eval build-out this session turned up. Nothing here is cluster-name
sensitive except where noted — treat every absolute path below as "wherever the new
cluster's fast shared filesystem is," not literally `/lustre/fsw/portfolios/adlr/...`.

## 0. What actually needs to survive the move

- **This git repo.** `git log` is the durable record of every script/entrypoint. If you
  can read this file, you have it.
- **Checkpoints** (`qad/checkpoints/`) — NOT in git (`.gitignore`'d, and would be
  enormous). These live only on the old cluster's filesystem. If they are not copied off
  before access is fully gone, every `qad-*` training run needs to restart from step 0.
- **Eval score results** (`qad/results/**/step_*.json`, `results/disagg/`,
  `results/ruler/`) — these ARE in git (see the `.gitignore` comment block "Eval results
  are versioned"), so `git log`/`git pull` recovers them. Raw generations
  (`*_samples_*.jsonl`) are NOT versioned by design (too large — see the same comment
  block) and only exist on the old filesystem.
- **HF cache** (`hf_cache/`) — re-downloadable, not urgent, but re-downloading Gemma-3-12b
  and Qwen3-8B from scratch is real wall-clock time; worth an `rsync` if there is a window
  where both clusters are reachable at once.
- **wandb run history** — synced to the wandb server already (that's what `WANDB_MODE`
  online means); nothing local to rescue.

## 1. Directory layout and the symlink gotcha

The root convention on the old cluster was `<fast-fs>/prefill_decode/`, holding this repo
plus every dependency overlay as SIBLING directories (not nested inside the repo):

```
prefill_decode/
  prefill-decode-shenanigans/   <- this git repo (qad/, evals/, notebooks/, latex/)
  containers/                   <- .sqsh images
  hf_cache/                     <- HF_HOME; token at hf_cache/token, mode 600
  lm_eval_overlay/               <- pip --target tree, see §3
  venv_overlay/                  <- pip --target tree, see §4
  nixl-nodeps/                   <- pip --no-deps tree, see §5
  checkpoints/ (or qad/checkpoints/, same thing via a relative default)
```

**`/lustre` was a plain symlink to `/scratch`** on the old cluster (`/lustre -> /scratch`,
confirm with `readlink -f` on the new one — it may not exist at all, or point somewhere
else). Two things broke repeatedly because of this and will break again on a new cluster
if this isn't checked first:

- Containers only mounted `/lustre:/lustre` — NOT `/scratch`, even though `/lustre` is
  just a symlink to it. A script that resolves its own physical path (`realpath`,
  `os.path.abspath(__file__)`, or anything that calls `getcwd()`) gets the `/scratch`-
  rooted form, which is invisible inside the container even though the file is right
  there. Fixed everywhere in this repo by using **logical** path resolution instead:
  `cd "$(dirname "${BASH_SOURCE[0]}")" && pwd` in bash (not `realpath "$0"`), and
  `os.path.join(os.environ.get("PWD", os.getcwd()), sys.argv[0])` in Python (not
  `__file__`, which CPython resolves to the physical path at interpreter startup before
  any of this repo's code runs). **On a new cluster, check what the fast filesystem's
  mount path actually is and whether the same symlink-vs-physical split exists** —  if
  it does not, none of this matters; if it does, the same idiom keeps working.
- `scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}'` (used everywhere to
  re-find a script's own path from inside a re-entered container) returns whatever path
  form the job was originally submitted with — so submitting via the `/lustre` form keeps
  that form throughout. Submit from the mounted path, not the physical one.

## 2. Container

`containers/nemo-26.02.sqsh` — a NeMo image with torch/vllm/transformers/numpy already
built for the cluster's GPU architecture (was aarch64/GB300 on the old cluster; **will be
a different image if the new cluster's GPUs/arch differ** — do not just copy the `.sqsh`,
rebuild or re-pull for the new architecture). `env.sh` in the old root still points at
`vllm-nightly-muse.sqsh`, which is stale — that image is from the prior b200 setup and was
kept only for reference; every script actually used this session pointed at
`nemo-26.02.sqsh`. Update `env.sh` (or write a new one) to reflect whatever image is
actually current before trusting it.

**GPU-allocated jobs hide `/opt/venv`.** The container ships extra packages (`wandb`,
`datasets`, etc.) under `/opt/venv/lib/python3.12/site-packages`, but the NVIDIA container
runtime hides/replaces that path the moment a job has a GPU allocation — confirmed via a
`PATH`/site-packages diff between a GPU job and a GPU-less one. `qad/sitecustomize.py`
(auto-imported via `site`, since `qad/` sits on `PYTHONPATH`) works around this: a
verbatim copy of `/opt/venv`'s site-packages lives on the fast filesystem
(`VENV_OVERLAY`), and `site.addsitedir(overlay)` **appends** it to `sys.path` rather than
prepending — prepending let the overlay's stale bundled `botocore` shadow the base
container's boto3-compatible one. If the new cluster's container doesn't have this same
hide-on-GPU-allocation quirk, `VENV_OVERLAY` can be left unset and `sitecustomize.py`
no-ops harmlessly (it checks `os.path.isdir` before doing anything).

**CPU-partition containers can resolve a DIFFERENT `transformers`/`huggingface-hub` than
GPU-partition ones**, for the same underlying reason (whether `/opt/venv` is visible).
Hit this diagnosing the RULER `vt_utils` bug: identical `PYTHONPATH`, but a CPU-partition
job saw `huggingface-hub==1.30.0` (from an overlay) conflict with the base container's
`transformers`, while GPU jobs never did. If a CPU-partition diagnostic script throws an
import error a GPU job doesn't, this is the first thing to check — don't assume it's a
real bug before checking whether it's this partition-visibility split.

**TLS**: `flashinfer`'s `cubin_loader.py` failed cert verification on the old cluster
until `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` were all pointed at
`/etc/ssl/certs/ca-certificates.crt` explicitly (see `qad/serving/run_nixl_server.sh`).
Worth setting proactively on a new cluster rather than waiting to hit the same failure.

## 3. lm_eval_overlay (evaluation)

```
pip install "lm_eval[api,math,ruler]" --target $LM_EVAL_OVERLAY
```
run **inside the container** (Python 3.12 ABI match — a login-node interpreter produces
a tree that fails on a compute node; the login node's own `python3` may not even have
torch, see §7). The plain-`pip install lm_eval` extras groups matter: `api` for
`LocalCompletionsAPI` (disagg/RULER serving both use it), `math` for minerva_math500, and
`ruler` for `wonderwords`/`nltk`, which RULER's `niah` tasks import directly and which a
bare `lm_eval` install does NOT pull in — this cost a full failed job before being caught
(`ImportError: Please install the wonderwords and nltk packages...`).

`PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:...` — overlay FIRST. Unlike `VENV_OVERLAY` above,
this one is meant to win: it's lm_eval itself, not a small set of missing extras.

**A dill/pyarrow bug lives here, not in this repo's code**, and will resurface on any
cluster with a similarly recent `pyarrow`+`dill` pair: `datasets.Dataset.from_list()`
(used by RULER's synthetic-task builders) always calls `generate_fingerprint()` at
construction time, which dill-hashes the whole object including the raw pyarrow Table —
and dill chokes on pyarrow's `MonthDayNano` class, which reports `__module__="builtins"`
(wrong, and the class is immutable so it can't be corrected). Reproduces on a trivial
`Dataset.from_list([{"a": 1}])`, independent of RULER's content. Confirmed this bug
persists across the whole `pyarrow` 21–25 range (whatever `datasets>=5.0` currently
requires), so pinning an older pyarrow does not fix it. `qad/eval/eval_ruler.py`'s
`_patch_datasets_fingerprint()` works around it by falling back to
`generate_random_fingerprint()` on hash failure — mirroring `datasets`' own existing
behavior for the exact same failure class in `update_fingerprint()` (used by
`.map()`/`.filter()`, which already has this fallback; `generate_fingerprint()`, used by
`__init__`, does not). If a fresh `pip install` on the new cluster pulls a `dill`/
`pyarrow` pair where this is fixed upstream, the patch is a silent no-op (it only
activates in the `except` branch) — safe to leave in either way.

**A second, unrelated bug also lives in lm_eval itself**: its custom `!function` YAML tag
resolution (`lm_eval/tasks/_yaml_loader.py:_load_module_with_cache`) does NOT do a normal
`import` — it computes the same dotted module name a normal import would use, checks
`sys.modules`, but only trusts what it finds there if the cached module carries a
`__mtime__` attribute matching the file's current mtime (a marker only that loader itself
ever sets). A plain `import lm_eval.tasks.ruler.vt_utils as m; m.some_func = patched` gets
silently discarded the moment any RULER task's YAML resolves `custom_dataset: !function`,
because the loader's freshness check fails and it re-execs the file fresh from disk,
overwriting `sys.modules[...]` with an unpatched copy — no error, no trace the patch ever
ran. This is exactly what happened when fixing `vt_utils.py`'s `generate_chains` (a real
upstream bug: its uniqueness-padding loop can leave the variable-name list length not a
clean multiple of the chunk size it's about to be sliced by, causing an `IndexError` at
long enough context). The fix in `eval_ruler.py`'s `_patch_ruler_vt_generate_chains()`
stamps `__mtime__` onto the patched module after patching it, so the loader's own
freshness check passes. **Any future monkeypatch of an lm_eval task-utility module (not
just vt_utils) needs this same `__mtime__` stamp, or it will silently not take effect the
same way** — this is a property of `_yaml_loader.py` itself, not of `vt_utils.py`
specifically, and will bite any task's util module the same way.

## 4. venv_overlay (training)

Verbatim copy of the container's own `/opt/venv/lib/python3.12/site-packages` (see §2 for
why this exists at all). Rebuild by launching a GPU-less container job, `cp -r` from
`/opt/venv` (visible there), and pointing `VENV_OVERLAY` at the copy. `qad/sitecustomize.py`
picks it up automatically via `PYTHONPATH` containing `qad/` — no other wiring needed.

## 5. nixl-nodeps (disaggregated serving)

```
pip install nixl nixl-cuNN --no-deps --target $NIXL_PREFIX
```
(exact package names/versions were whatever matched the CUDA build on the old cluster —
re-resolve against the new cluster's CUDA version, don't copy the tree). `--no-deps` is
not optional: without it, pip pulls a second `torch` into the overlay, which shadows the
container's CUDA-matched build and breaks `vllm._C` with an undefined `at::TensorBase`
symbol. Only `run_nixl_server.sh` (disaggregated prefill/decode serving) puts this on
`PYTHONPATH`, and it goes there via prepend (the only overlay that does — see
`evals/bin/install_overlays.sh`'s comment for why).

`kv_buffer_device=cuda` (not `cpu`) — this was flipped mid-session after a *previous*
cluster (b200) had needed `cpu` staging because UCX there lacked CUDA support. **This is
GPU/fabric-specific, not a fixed setting** — confirmed on this GB300 cluster that `cpu`
staging actively breaks under `TP>1` (host-RAM OOM) while `cuda` works cleanly. On a new
cluster, treat this as re-measurable, not as "already solved" — check UCX's CUDA support
on the new fabric before assuming either value.

`TP=2` (tensor parallelism) is the default in `run_nixl_server.sh`, chosen to avoid
sitting a 4-GPU floor half-idle (see §7) — re-validate this is still the right trade-off
if the new cluster's GPU-per-node count or interconnect differs.

## 6. RULER-specific data

`lm_eval[ruler]`'s NIAH tasks download essay text and auto-fetch `nltk`'s `punkt_tab`
tokenizer data on first use — this writes into a non-persistent, container-internal
location under `--no-container-mount-home`, meaning **every job re-downloads it**. Never
became a real problem (small, fast) but on a new cluster with slower/metered egress this
is worth pointing `NLTK_DATA` at a persistent shared path instead. `ruler_qa_squad`/
`ruler_qa_hotpot` fetch their raw JSON over plain HTTP directly (not through
`huggingface_hub`), so `HF_HUB_OFFLINE=1` does NOT make a RULER job network-independent
even once the HF-hosted pieces are cached.

## 7. Cluster-specific numbers to re-measure, not copy

Everything below was true for `aws-cmh-slurm-1` specifically and needs re-checking, not
blind reuse:

- **QOS `MinTRES gres/gpu=4`**: this cluster refused any job requesting fewer than 4 GPUs
  outright — GPU counts and `--gpus-per-node` throughout this repo's `bin/` scripts are
  built around that floor. A new cluster may have a different (or no) floor.
- **Login-node `python3` has nothing installed** (it's a bare `uv` venv) — every diagnostic
  that needs `torch`/`transformers`/etc. has to run inside a container job (CPU-partition
  for cheap checks, GPU for anything touching CUDA), never bare on the login node. Verify
  this holds on the new cluster before assuming it.
- **QOS names** (`normal`, `interactive`, `short`, `cpu`, ...) — `sacctmgr show qos` to get
  the real list on the new cluster; `interactive` (higher priority, meant for short
  verification runs, not bulk sweeps) was load-bearing this session for jumping a
  saturated queue during a fix-verification loop.
- **`--chain N --dependency=singleton`** (in `qad/bin/run_qad.sh`) works around the
  cluster's 4-hour wall-clock cap by submitting N jobs that share a job name and run
  strictly one after another, each resuming via `--resume auto` off the last one's
  `state/` checkpoint (written every `--save-every=100` steps). A chain job that finds the
  run already at its target step exits immediately doing nothing, so it's safe to
  over-provision. If the new cluster's wall-clock cap differs, this mechanism still
  works, it just needs a different N to reach the same total step count.

---

# 8. The actual migration to `oci-jhb-slurm-1` (2026-09-10)

Everything above is the general guide. This section is the concrete record of doing it,
including the places where the guide above turned out to be **wrong for this cluster**.
Read §8.3 before touching the eval overlays — that one is a new failure mode.

## 8.0 Root layout

```
/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/
  prefill-decode-shenanigans/   <- this repo
  containers/nemo-26.02.sqsh    <- 35 GB, see §8.2
  hf_cache/                     <- HF_HOME, 63 GB, token at hf_cache/token (600)
  lm_eval_overlay/              <- EVAL overlay, 772 MB, see §8.3
  nixl_nodeps/                  <- serving overlay, 113 MB, see §8.4
  chunk_cache/                  <- tokenized Tulu chunks, shared across formats
  (no venv_overlay/ -- deliberately absent, see §8.3)
```

Account `coreai_psx_qad`. Partitions: `batch` (4 h), `batch_long` (7 d), `cpu` (7 d),
`cpu_datamover`. QoS: `normal`, `short`, `interactive`, `free`, `cpu-*`. Nodes are
**4-GPU** GB300 trays with 144 CPUs / 942 GB RAM, so `--nodes 2` = the world-8 default.

## 8.1 The `/lustre` symlink question is GONE here — use physical paths

`/lustre` is a root-level symlink to `/scratch`, **and** `…/coreai/users/apanferov` is a
second symlink to `…/coreai/projects/coreai_psx_qad/users/apanferov`, so the same
directory has three spellings. Rather than replay §1's logical-vs-physical dance, this
migration uses the **physical `/scratch` path everywhere** and mounts
`--container-mounts=/scratch:/scratch,/lustre:/lustre`. `getcwd()`/`realpath` then agree
with the mount, so §1's whole bug class ("the file is right there but invisible inside
the container") cannot occur. Keep it that way.

## 8.2 Container

Stock `nvcr.io/nvidia/nemo:26.02` from NGC (anonymous pull works; `26.02` still exists and
ships an **arm64** variant — verify with the NGC API before assuming a tag). Imported by a
**CPU-partition** job with `ENROOT_TEMP_PATH=/raid/scratch` (this cluster's node-local
21 TB scratch, cleared between jobs). Both constraints from `evals/bin/import_container.sh`
still hold and still fail silently if ignored. Import took ~7 min → 35 GB `.sqsh`.

Contents that matter: Python 3.12.3, torch 2.10 (CUDA **13.0**), transformers 4.57.6,
datasets 3.1.0, huggingface_hub **0.36.2**, wandb 0.25.0, vllm 0.14.2.dev0, triton 3.5.0.

**`git submodule update --init third_party/Liger-Kernel` is mandatory and is NOT
optional bookkeeping.** A fresh clone leaves it empty; `training/qad.py` puts
`third_party/Liger-Kernel/src` on `sys.path` directly, so the run dies 8 ranks deep with
`ModuleNotFoundError: No module named 'liger_kernel'` after already burning the model
load. This cost one job here.

## 8.3 Overlays: THREE separate trees, never one

The overlays exist for different consumers with **opposite sys.path precedence**, so they
must stay separate trees. Merging them forces one precedence on both and reintroduces the
shadowing failures below.

| tree | consumer | precedence | needed here? |
|---|---|---|---|
| `venv_overlay/` | training (`run_qad.sh`) | **appended** via `sitecustomize.py` | **NO** — see below |
| `lm_eval_overlay/` | eval *driver* | **prepended** | yes |
| `nixl_nodeps/` | prefill/decode *servers* | **prepended** | yes |

**`venv_overlay` is not needed on this cluster.** §2's premise — that the NVIDIA container
runtime hides `/opt/venv` the moment a job holds a GPU — does **not** reproduce here.
Verified directly under a 4-GPU allocation: `sys.executable` is `/opt/venv/bin/python3`
and `wandb`/`datasets` (the packages §2 says live only there) import fine. The directory
is deliberately left absent; `sitecustomize.py` checks `os.path.isdir` and no-ops. It is
still wired into `run_qad.sh`, so if a future image *does* hide `/opt/venv`, populating
that one path is the whole fix.

**`lm_eval_overlay` MUST be installed with a `huggingface_hub<1.0` constraint. This is new
and it is the one that will bite.**

```bash
pip install "lm_eval[api,math,ruler]" "huggingface_hub<1.0" --target $LM_EVAL_OVERLAY
```

run **inside the container**. Without the constraint, lm_eval resolves
`huggingface-hub==1.30.0`, which is prepended ahead of the container's 0.36.2 and makes
**`transformers` itself fail to import**:

```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this
module, but found huggingface-hub==1.30.0.
```

That breaks every RULER job, because RULER's synthetic tasks need `transformers` to size
their haystacks to each target length. §3 saw a milder version of this (hub 1.24 breaking
`vllm serve`, worked around by `eval_disagg.py` stripping the overlay for the *servers*);
the constraint fixes the *driver* too, and pins the overlay to the container's exact
0.36.2. The extras still matter for the same reasons as §3: `api`, `math`, and `ruler`
(the last supplies `wonderwords`/`nltk`).

The §3 **dill/pyarrow `MonthDayNano` bug is still live** — reproduced here on dill 0.4.1 +
pyarrow 25.0.1 + datasets 5.0.1, with a bare `Dataset.from_list([{"a": 1}])` raising
`PicklingError: Can't pickle <class 'MonthDayNano'>`. `eval_ruler.py`'s
`_patch_datasets_fingerprint()` is therefore load-bearing, not vestigial. The `__mtime__`
stamping rule for any lm_eval task-utility monkeypatch (§3) likewise still applies.

## 8.4 nixl

```bash
pip install nixl nixl-cu13 --no-deps --target $NIXL_PREFIX
```

`nixl-cuNN` in §5 was a placeholder — it 404s on PyPI. The CUDA-matched name is
**`nixl-cu13`** (container is CUDA 13.0). `--no-deps` is still mandatory. The import gate
passes: with `nixl_nodeps` on `PYTHONPATH`, `import nixl._api`, `import vllm` and
`import vllm._C` all succeed in one interpreter (no undefined `at::TensorBase`).

### 8.4.1 CUDA nixl: SOLVED — the eval container ships `UCX_TLS=tcp`

Disaggregated serving works, with `kv_buffer_device=cuda`. The 1P1D gate PASSES:

```
A(disagg W4A4->W4A16) identical to B(homogeneous W4A16): 0/4
A(disagg W4A4->W4A16) identical to C(homogeneous W4A4) : 0/4
VERDICT: PASS -- KV crossed.
```

**Root cause of "UCX CUDA support was not found".** Nothing is missing from the image.
The CUDA module loads correctly (`loaded .../libuct_cuda.so.0.0.0`, `dmabuf is supported
on cuda device 0`, `multi-node NVLINK support is enabled`). The problem is that the
vLLM container **ships `UCX_TLS=tcp` in its own environment**, and a tcp-only transport
list deselects every CUDA transport. UCX then closes the memory domain —
`closing md cuda_cpy because it has no selected transport resources` — after which
VRAM looks like host memory (`VRAM memory is detected as host by UCX`) and
`register_kv_caches` dies with `NIXL_ERR_BACKEND`.

Measured with the minimal VRAM-registration gate (allocate a CUDA tensor, create a
UCX-backed nixl agent, `register_memory(..., "VRAM")`):

| env | result |
|---|---|
| *(inherited: `UCX_TLS=tcp`)* | **FAIL** `NIXL_ERR_BACKEND` |
| `UCX_MEMTYPE_CACHE=n` alone | **FAIL** (red herring — not a memtype-cache bug) |
| `UCX_TLS=cuda_copy,cuda_ipc,tcp,self,sm` | **OK** |
| `UCX_TLS=all` | **OK** |
| `UCX_TLS=all` + `UCX_NET_DEVICES=eth0` | **OK** |

So the fix is two exports, both now defaults in `serving/run_nixl_server.sh`:

- `UCX_TLS=all` (was already the default there — it is what makes CUDA work, and the
  reason must not be lost: it exists to override the image's tcp-only value).
- `UCX_NET_DEVICES=eth0` (**changed from `all`**). This cluster's `rdma_vf_rail0..3`
  carry ONLY IPv6 addresses; with `all`, UCX picks a rail and dies at
  `bind(addr=fdcd:...%0:0) failed: Cannot assign requested address`. `eth0` is the
  node's only IPv4 device, and is what `NCCL_SOCKET_IFNAME` uses too.

**Use the vLLM container for disaggregated serving, not NeMo.** `nemo-26.02` additionally
suffers the two-UCX-cores conflict from `run_nixl_ucx_diag.sh` (its HPCX UCX plus the pip
wheel's vendored copy, mangled sonames), where the CUDA transport registers into one
core's registry while nixl's `ucp_context` belongs to the other. Under NeMo the three
`UCX_NET_DEVICES` values fail three different ways (IPv6 bind error / no-route /
segfault) and `kv_buffer_device=cpu` does not rescue it. The vLLM image has exactly ONE
UCX (the wheel's) and no such conflict. `bin/run_eval_disagg.sh`, `bin/run_eval_ruler.sh`
and `diagnostics/run_kv_verify.sh` now default to `containers/vllm-nightly.sqsh`.

`NIXL_PREFIX` now defaults to the container's own `/usr/local/lib/python3.12/dist-packages`
because the vLLM image ships nixl 1.3.2. Do not prepend the separate `nixl_nodeps` tree in
that container — a second nixl shadows the one vLLM was built against. `nixl_nodeps`
remains only for an image lacking nixl entirely (nemo-26.02).

**Fixed along the way:** `diagnostics/verify_kv_transfer.py` resolved its driver as
`_QAD/eval_disagg.py` with `_QAD = Path(__file__).parent` — i.e. `qad/diagnostics/`,
while the file actually lives at `qad/eval/eval_disagg.py`. The gate died with
`can't open file .../diagnostics/eval_disagg.py` before running a single stack, so it had
been broken since that file moved.

### 8.4.2 The eval container and its overlay (and why there are now two overlays)

`containers/vllm-nightly.sqsh` — `docker://vllm/vllm-openai:nightly`, arm64, 17.7 GB.
`nightly` is a MOVING tag and enroot cannot pin by digest (its URI scheme is
`docker://[USER@][REGISTRY#]IMAGE[:TAG]`, so `@` is the USER separator and
`image@sha256:...` misparses). Provenance is therefore recorded beside the image in
`containers/vllm-nightly.provenance.txt`:

```
index_digest:  sha256:96b234afa2867031ad0a226b149a06483861e613e3a3083da53398d347b5ffbd
arm64_digest:  sha256:9cbd898be1bf2d258767f923d2ee4fa6bc32eb95db3b8c97d3a7d43d37e7873c
sqsh_sha256:   c40a305ddf03c99c94917b2fc650525433a324d1e9cc45aac8b791cc10c1135e
imported_utc:  2026-09-10T12:24:09Z
```

The digest was re-queried after the pull and matched, so the tag did not move mid-import.
Re-fetch that exact image with `skopeo copy docker://vllm/vllm-openai@<index_digest>`.
Contents: vLLM 0.28.1rc1.dev628, torch 2.13.0+cu130, **transformers 5.17.0**,
huggingface_hub 1.30.0, nixl 1.3.2. No HPCX, no system `ucx_info`.

**There are now TWO eval overlays, and they are not interchangeable** — this is the
strongest form of the §8.3 rule. Each must be pip-installed *inside the container it will
be used with*, because the correct `huggingface_hub` pin is opposite in the two:

| overlay | container | transformers | hub | install |
|---|---|---|---|---|
| `lm_eval_overlay` | `nemo-26.02` (training-side / non-disagg evals) | 4.57.6 | **0.36.2** | `pip install "lm_eval[api,math,ruler]" "huggingface_hub<1.0" --target ...` |
| `lm_eval_overlay_vllm` | `vllm-nightly` (disaggregated evals, RULER) | 5.17.0 | **1.31.0** | `pip install "lm_eval[api,math,ruler]" --target ...` (no constraint) |

Using the NeMo overlay inside the vLLM container fails immediately with
`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'` — hub 0.36.2
is too OLD for transformers 5.17.0. Conversely the unconstrained overlay inside NeMo
pulls hub 1.30.0, which is too NEW for transformers 4.57.6. Same package, opposite pins.

## 8.5 Scaling: §7's table is too conservative for GB300

`MinTRES gres/gpu=4` is not the binding constraint here; whole nodes are, and they are
4-GPU. The README's scaling table assumes lower-memory GPUs than GB300:

- **gemma-3-12b-it single-master fits at world 8 / DP 8 / mbs 8 with no PP and no
  `--recompute-wq`** — measured, not assumed. The README's "12B → world 16, DP 16"
  is unnecessary here.
- This matters because **DP width must be held constant across runs**: gradient clipping
  uses the RMS of per-rank norms, so a run at DP 16 is not numerically comparable to one
  at DP 8. Every run in this sweep is DP 8. If a future model genuinely will not fit,
  reach for `--pp 2` (which adds nodes while holding DP) rather than more DP.

## 8.6 Warm the HF cache BEFORE the first training job

`run_qad.sh` defaults `HF_HUB_OFFLINE=1` for the reason its own comment gives: 8 ranks
fetching a *sharded* checkpoint concurrently race, and one loses with
`OSError: … does not appear to have a file named model-0000N-of-0000M.safetensors` even
though the cache is complete. So on-demand download from inside training is exactly what
that flag exists to prevent — download first, with plain `hf download <repo>` (do **not**
pass `--exclude`: its globs are swallowed as positional FILENAMES, and the command then
downloads nothing and still exits 0). Gemma-3 is gated and needs `hf auth login` first.
Verify by counting `*.safetensors` per repo, not by exit code.

Datasets: one run with `HF_HUB_OFFLINE=0` warms the `datasets` arrow cache for
`allenai/tulu-3-sft-mixture`; everything after can stay offline.

## 8.7 Chunk cache

`--chunk-cache-dir` is keyed on (model, split, tokens, seq, rank, world) and **not** on
the quantizer, so all formats of one model share one tokenization — and chained jobs stop
re-tokenizing on every resume. `build_chunks` now writes via a unique temp file +
`os.replace` rather than `torch.save` straight onto the target, because concurrent formats
legitimately race on that path and an in-place save is not atomic (a reader mid-write got
a truncated pickle).

## 8.9 RULER needs `baber/paul_graham_essays` pre-cached, or every job dies

`bin/run_eval_ruler.sh` exports BOTH `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1` (so
hundreds of concurrent jobs do not earn a Hub 429). RULER's NIAH tasks build their haystack
from the HF-hosted dataset `baber/paul_graham_essays` at TASK-BUILD time, so if it is not
already in `hf_cache/`, every NIAH task raises

```
ConnectionError: Couldn't reach 'baber/paul_graham_essays' on the Hub (OfflineModeIsEnabled)
```

and the job exits 1 — after already paying ~170 s to bring the 1P1D stack up. This killed
**222 of 224** submitted RULER jobs on the first real run. Warm it once:

```bash
HF_HOME=<hf_cache> HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
  python3 -c "import datasets; datasets.load_dataset('baber/paul_graham_essays', split='train')"
```

218 rows; lands in both `hub/datasets--baber--paul_graham_essays` and
`datasets/baber___paul_graham_essays`. Verified afterwards that all NIAH plus
vt/cwe/fwe/qa tasks build with offline mode ON.

**Why the bring-up checks missed it:** every RULER validation run passed
`HF_HUB_OFFLINE=0` explicitly, so it fetched the essays live and never exercised the
offline path the real entrypoint uses — and nothing persisted the download, so the cache
stayed empty and the next (offline) job still failed. **Validate an eval entrypoint with
the SAME offline flags it sets itself, not with the network open.**
`ruler_qa_squad`/`ruler_qa_hotpot` are unaffected either way: they fetch raw JSON over
plain HTTP, which the HF offline flags do not gate (§6).

## 8.8 Eval autosubmit: three silent-nothing traps

`cluster_scripts/submit_missing_evals.py` + `autoeval_watch.sh` are the eval entrypoint.
All three failures below exit 0 and print a plausible "0 points" — none of them errors.

1. **The family name is no longer the checkpoint prefix.** `PREFIX = "qad3x"` is now a
   single constant for every family; a separate `gemma3-` checkpoint namespace was a relic
   of the early Gemma ablations. Previously the family key doubled as the prefix, so this
   cluster's Gemma sweep (trained under run_qad.sh's default `RUN_PREFIX=qad3x`) was
   globbed as `gemma3-google-gemma-3-*`, matched nothing, and reported
   `would submit 0 ... covering 0 point(s)` for all 12 runs. A family now selects MODELS
   AND MODES only.

2. **Stale results from a previous training run block the new one.** Result tags are
   `<prefix>-<model>-<quant>-<hash>`, and the hash covers only quantizer PARAMS — so a
   retrain of the same format produces the IDENTICAL tag. The old cluster's
   git-versioned `results/` therefore occupied every grid point for Qwen, and the gap scan
   found nothing to do for all 12 fresh Qwen runs. Old-cluster RULER dirs for
   nvfp4/nvfp4prefill/nvfp4decode were deleted for this reason (Qwen RULER coverage went
   104 -> 118 points). **After any retrain, check `results/` for colliding tags before
   trusting a "no gaps" scan.** Non-RULER (`results/disagg/`) trees were left intact
   deliberately — this sweep is RULER-only.

3. **`DEFAULT_EXCLUDED_FORMATS` excludes exactly nvfp4 / nvfp4prefill / nvfp4decode**
   from the DISAGG sweep and routes them to RULER instead. For a RULER-only sweep this is
   what you want and needs no change: a bare `autoeval_watch.sh` submits RULER only. If
   gsm8k/math500/mmlu_pro are ever wanted for these formats, pass
   `--formats nvfp4 nvfp4prefill nvfp4decode` explicitly rather than editing the set.

4. **RULER had no submit-time dedup, and duplicated the whole sweep.** `ruler_inflight()`
   is deliberately NOT ledger-based: it reads each live `qad-ruler` job's stdout banner to
   learn which step it covers, and its docstring accepts that PENDING jobs are invisible
   as "bounded, self-healing". That assumption fails completely when jobs CANNOT START --
   under this cluster's 9-hour maintenance reservation all 224 submitted RULER jobs sat
   PENDING, so nothing was ever in flight and a second run re-submitted the entire grid
   (observed: 448 queued, exactly 2x224). `autoeval_watch.sh` calls `--apply` every
   `INTERVAL` (default 120 s), so it would have re-submitted ~224 jobs PER POLL for the
   whole window. Fixed by recording RULER submissions in the shared ledger (keyed
   `tasks="ruler"`) and subtracting it alongside the banner scan; the ledger records at
   SUBMIT time, so PENDING is covered. It also turned out `ruler_inflight()` never worked
   even for RUNNING jobs: it read `squeue -o "%o"`, but `%o` is the COMMAND (the
   run_eval_ruler.sh path), not StdOut -- so it opened the SCRIPT, searched it for a run
   banner, found none, and returned `{}` unconditionally. Now uses `-O StdOut`.
   The `autoeval_watch.sh` docstring claim "SAFE TO RUN TWICE ... keeps a ledger" was
   therefore true only of the DISAGG path.

Also fixed: `autoeval_watch.sh`'s `EVAL_JOBS` regex omitted `qad-ruler`, so every running
RULER job counted as a TRAINING job — `train` never reached 0 and the watcher would never
exit, while under-counting `eval_jobs`.
