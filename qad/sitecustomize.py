"""Auto-imported by Python's site module at interpreter startup (qad/ is on
PYTHONPATH, so this is always found). Appends VENV_OVERLAY to sys.path rather than
letting it sit ahead of the container's own site-packages via PYTHONPATH.

Why: on this cluster, a GPU-allocated container hides /opt/venv (the NeMo image's
real site-packages -- wandb, datasets, etc. live only there), so training pulls a
copy of it onto Lustre instead (VENV_OVERLAY). But that copy duplicates packages the
base container ALSO ships (e.g. accelerate), sometimes at different versions, and a
plain PYTHONPATH prepend makes the copy win every time -- which is how a stale
bundled botocore ended up shadowing the base container's boto3-compatible one.
site.addsitedir() appends instead, so the base container's own packages resolve
first and the overlay only fills genuine gaps.
"""
import os
import site

overlay = os.environ.get("VENV_OVERLAY")
if overlay and os.path.isdir(overlay):
    site.addsitedir(overlay)
