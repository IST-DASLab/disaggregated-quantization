"""Render setup metadata / historical attention measurements; see recorded_tables.json."""
import argparse
import json
from pathlib import Path
from common import row


def render(name):
    data = json.loads(Path(__file__).with_name("recorded_tables.json").read_text())[name]
    lines = [r"\begin{tabular}{" + data["columns"] + "}"]
    for entry in data["rows"]:
        lines.append(row(entry["cells"]) + entry.get("after", "") if "cells" in entry else entry["rule"])
    return "\n".join(lines + [r"\end{tabular}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("hyper", "parallel", "models", "attn-backend"))
    print(render(parser.parse_args().name))
