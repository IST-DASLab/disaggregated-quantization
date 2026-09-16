"""The in-container command is a double-quoted bash -c "..." string spanning ~80 lines.
An unescaped " ANYWHERE inside it -- including in a comment -- terminates the string
early, mangles the srun invocation, and produces a job that exits 0 with EMPTY logs.
That is exactly how job 365035 'COMPLETED' in 44s having trained nothing.

Check every line of that block for unescaped double quotes.
"""
import re
import sys

import pathlib; p = str(pathlib.Path(__file__).resolve().parent.parent / "bin" / "run_qad.sh")
lines = open(p).read().splitlines()

# The block runs from the `bash -c "` that opens it to the closing `" -- "${EXTRA_ARGS...`
start = next(i for i, l in enumerate(lines) if re.search(r'bash -c "\s*$', l))
end = next(i for i, l in enumerate(lines) if i > start and l.strip().startswith('" --'))
print(f"bash -c block: lines {start + 1}..{end + 1}")

bad = []
for i in range(start + 1, end):
    line = lines[i]
    # Every " inside the block must be backslash-escaped.
    for m in re.finditer(r'"', line):
        if m.start() == 0 or line[m.start() - 1] != "\\":
            bad.append((i + 1, line.strip()[:96]))
            break

if bad:
    print(f"\nUNESCAPED double quotes on {len(bad)} line(s) -- these break the string:")
    for n, t in bad:
        print(f"  {n}: {t}")
else:
    print("\nPASS: no unescaped double quotes inside the bash -c string")
sys.exit(1 if bad else 0)
