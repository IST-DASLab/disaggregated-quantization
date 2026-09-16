"""CPU-only regression checks for ODP data selection, tables and the schematic."""

import ast
import csv
import importlib.util
import json
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "qad/kernels/prefill"


def read_csv(name):
    with (KERNELS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


class ODPReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_csv("offload_prefill.csv")
        cls.latencies = {(r["model"], r["quant"], r["mode"], int(r["seq_len"])):
                         float(r["latency_ms"]) for r in cls.rows}

    def test_protocol_records_match_main_csv_and_byte_accounting(self):
        self.assertEqual(len(self.rows), len(self.latencies))
        seen = set()
        for row in read_csv("odp_prefill.csv"):
            self.assertEqual(row["protocol"], "odp_carveout_cold_v1")
            key = (row["model"], row["quant"], row["mode"], int(row["seq_len"]))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(float(row["latency_ms"]), self.latencies[key])
            self.assertEqual(int(row["slot_bytes"]), int(row["restore_bytes"]))
            self.assertEqual(int(row["ssd_bytes_per_request"]),
                             int(row["prefill_bytes"]) + int(row["restore_bytes"]))
        self.assertEqual(seen, {key for key in self.latencies if key[2] == "zero-ssd"})

    def test_notebook_uses_llama_cpp_ttft_and_reported_ranges(self):
        nb = json.loads((ROOT / "notebooks/plots.ipynb").read_text())
        source = next("".join(c.get("source", [])) for c in nb["cells"]
                      if "def load_gsq_rco_prefill(" in "".join(c.get("source", [])))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "load_gsq_rco_prefill")
        ns = {"Path": Path, "np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "pareto_loader", "exec"), ns)
        path = ROOT / "notebooks/data/qwen3_8_27b_llama_cpp_ttft.csv"
        curves = ns["load_gsq_rco_prefill"](path)
        self.assertEqual(set(curves), {"baseline", "odp"})
        self.assertEqual(set(curves["baseline"]), {128 * 2**i for i in range(9)})
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            length = int(row["seq_len"])
            for mode in ("baseline", "odp"):
                self.assertEqual(curves[mode][length], tuple(float(row[f"{mode}_{field}"])
                    for field in ("ttft_ms", "ttft_min_ms", "ttft_max_ms")))
            self.assertAlmostEqual(float(row["baseline_ttft_ms"]) / float(row["odp_ttft_ms"]),
                                   float(row["speedup_x"]), delta=0.00005)
            self.assertEqual((int(row["reps"]), int(row["warmup"])), (3, 1))
            self.assertEqual(row["prompt_cache"], "off")
        self.assertEqual(curves["baseline"][8192][0], 12273.2)
        self.assertEqual(curves["odp"][8192][0], 6902.1)
        speedups = {n: curves["baseline"][n][0] / curves["odp"][n][0]
                    for n in curves["baseline"]}
        self.assertEqual(max(speedups, key=speedups.get), 8192)
        self.assertEqual(min(n for n, ratio in speedups.items() if ratio > 1), 4096)
        self.assertEqual(f'{speedups[8192]:.2f}', '1.78')
        self.assertNotIn('"offload_prefill.csv"', source)
        self.assertIn('"SSD loading latency"', source)
        self.assertIn('load_floor_zero_ssd.csv', source)
        self.assertIn('lengths = sorted(n for n in prefill_curves["baseline"] if n >= 1024)', source)

    def test_timeline_includes_first_read_and_matches_total(self):
        path = ROOT / "notebooks/schematics/fig_odp_timeline.py"
        spec = importlib.util.spec_from_file_location("odp_timeline_test", path)
        module = importlib.util.module_from_spec(spec)
        previous_path = sys.path.copy()
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous_path
        self.assertAlmostEqual(module.ROWS[0][2], module.T_READ)
        self.assertAlmostEqual(module.ROWS[0][3], module.T_READ)
        self.assertAlmostEqual(module.ROWS[-1][4], module.MS_OFFLOAD)
        for previous, current in zip(module.ROWS, module.ROWS[1:]):
            self.assertAlmostEqual(previous[4], current[3])
        fig = module.build()
        try:
            bar = next(p for ax in fig.axes for p in ax.patches
                       if p.get_gid() == "decode-restore")
            self.assertAlmostEqual(bar.get_width(), 2 * module.T_READ)
            self.assertAlmostEqual(bar.get_x(), module.ROWS[-1][3])
            self.assertLessEqual(bar.get_x() + bar.get_width(), module.ROWS[-1][4])
            wait = next(p for ax in fig.axes for p in ax.patches
                        if p.get_gid() == "decode-restore-wait")
            self.assertAlmostEqual(wait.get_x(), bar.get_x() + bar.get_width())
            self.assertAlmostEqual(wait.get_x() + wait.get_width(), module.ROWS[-1][4])
        finally:
            module.plt.close(fig)

    def test_appendix_speed_table_matches_measurements(self):
        section = (ROOT / "notebooks/tables/prefill-per-model.tex").read_text()
        table_rows = [line for line in section.splitlines()
                      if line.startswith(("Qwen3-", "Qwen3.8-", "Gemma-3-"))]
        self.assertEqual(len(table_rows), 9)
        models = {r["model"] for r in self.rows}
        for line in table_rows:
            cells = [part.strip() for part in line.split("&")]
            label = cells[0].split("$")[0]
            if label.startswith("Qwen"):
                model = "Qwen/" + label
            else:
                model = next(m for m in models
                             if m.lower().startswith("google/" + label.lower()))
            for offset, length in ((1, 16384), (4, 32768)):
                baseline = self.latencies[model, "bf16", "resident", length]
                self.assertEqual(cells[offset], f"{baseline:.0f}")
                for column, mode in ((offset + 1, "resident"), (offset + 2, "zero-ssd")):
                    expected = baseline / self.latencies[model, "nvfp4", mode, length]
                    actual = float(re.match(r"[\d.]+", cells[column]).group())
                    self.assertTrue(math.isclose(actual, expected, abs_tol=0.005),
                                    (model, length, mode, actual, expected))


if __name__ == "__main__":
    unittest.main()
