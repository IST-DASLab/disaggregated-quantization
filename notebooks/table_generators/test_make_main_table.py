"""Offline checks for RULER averaging, weight residency and the DH/PH table."""

import contextlib
import io
import unittest
from unittest.mock import patch, mock_open

import make_main_table as table


class RulerAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.lengths = [4096, 8192, 16384, 32768]
        self.curves = {
            length: {step: 10.0 * (i + 1) for step in table.TAIL_STEPS}
            for i, length in enumerate(self.lengths)
        }
        self.ns = {
            "RULER_LENGTHS": self.lengths,
            "ruler_models": {"model": {"method": self.curves}},
            "ruler_baselines": {"model": dict(zip(self.lengths, [20., 40., 60., 80.]))},
            "ruler_avg_curve": lambda model, curves: {
                step: sum(curves[length][step] for length in self.lengths) / 4
                for step in set.intersection(*(set(curves[length]) for length in self.lengths))
            },
        }
        self.incomplete = []

    def test_equal_length_weight_and_fixed_checkpoints(self):
        for curve in self.curves.values():
            curve[0] = 0.0
            curve[2500] = 100.0
        self.curves[65536] = dict.fromkeys(table.TAIL_STEPS, 0.0)
        self.assertEqual(table.ruler_accuracy(self.ns, "model", "method", self.incomplete), 25.)
        self.assertEqual(table.ruler_accuracy(self.ns, "model", None, self.incomplete), 50.)
        self.assertFalse(self.incomplete)

    def test_missing_length_is_not_a_partial_mean(self):
        del self.curves[4096]
        self.assertIsNone(table.ruler_accuracy(self.ns, "model", "method", self.incomplete))
        self.assertIn("4K step(s)", self.incomplete[0][2])

    def test_missing_or_invalid_step_is_not_replaced_by_later_step(self):
        for value in (None, float("nan")):
            with self.subTest(value=value):
                self.curves[8192][2500] = 100.
                if value is None:
                    self.curves[8192].pop(2250, None)
                else:
                    self.curves[8192][2250] = value
                self.assertIsNone(table.ruler_accuracy(self.ns, "model", "method", self.incomplete))
                self.assertIn("8K step(s) 2250", self.incomplete[-1][2])

    def test_missing_baseline_is_reported(self):
        del self.ns["ruler_baselines"]["model"][32768]
        self.assertIsNone(table.ruler_accuracy(self.ns, "model", None, self.incomplete))
        self.assertEqual(self.incomplete[0][1], "BF16")
        self.assertIn("32768", self.incomplete[0][2])


class PrefillSpeedupTests(unittest.TestCase):
    def test_uses_zero_ssd_without_block_size_metadata(self):
        rows = ["model,quant,mode,seq_len,latency_ms"]
        for model in (table.QWEN_BIG, table.GEMMA_BIG):
            for quant, mode, ms in [("bf16", "resident", 300),
                                    ("nvfp4", "resident", 150),
                                    ("nvfp4", "ssd", 100),
                                    ("nvfp4", "zero-ssd", 200)]:
                rows.append(f"{model},{quant},{mode},{table.SEQ},{ms}")
        with patch("builtins.open", mock_open(read_data="\n".join(rows))):
            result = table.prefill_speedups()
        self.assertEqual(result, {m: (2.0, 1.5) for m in (table.QWEN_BIG, table.GEMMA_BIG)})


class TableOutputTests(unittest.TestCase):
    def test_both_workloads_and_todo_without_changing_cost_columns(self):
        def accuracy(sizes, method, workload="decode"):
            if workload == "decode":
                return 60.
            return 80. if sizes == table.QWEN else None

        pre = {model: (1.5, 1.4) for model in (table.QWEN_BIG, table.GEMMA_BIG)}
        dec = {model: {row[3]: 2. for row in table.ROWS} for model in pre}
        output = io.StringIO()
        with patch.object(table, "accuracies", return_value=(accuracy, [])), \
             patch.object(table, "prefill_speedups", return_value=pre), \
             patch.object(table, "decode_speedups", return_value=dec), \
             patch.object(table, "param_counts", return_value=(1_000_000_000, 0)), \
             contextlib.redirect_stdout(output):
            table.main()
        text = output.getvalue()
        self.assertIn(r"\begin{tabular}{l|ccccc|ccccc}", text)
        self.assertEqual(text.count(r"\makecell{Acc.\\PH}"), 2)
        rows = [line for line in text.splitlines() if " & " in line
                and "makecell" not in line and "multicolumn" not in line]
        self.assertEqual(len(rows), len(table.ROWS))
        for row in rows:
            cells = row.split(" & ")
            self.assertEqual(len(cells), 11)
            self.assertEqual(cells[4:6], ["60.0", "80.0"])
            self.assertEqual(cells[9], "60.0")
            self.assertTrue(cells[10].startswith(r"\textcolor{red}{TODO}"))

        # ODP retains full-disaggregation accuracy but only decode-weight residency.
        # Check both bitwidths and both family columns in the generated table.
        for i, spec in enumerate(table.ROWS):
            if spec[0] != "+ODP":
                continue
            cells = rows[i].split(" & ")
            full = rows[i - 1].split(" & ")
            weight_only_index = next(j for j, row in enumerate(table.ROWS)
                                     if row[0].endswith("weight-only") and row[4] == spec[4])
            weight_only = rows[weight_only_index].split(" & ")
            for col in (3, 8):
                self.assertEqual(cells[col], weight_only[col])
                self.assertAlmostEqual(float(full[col]) - float(cells[col]),
                                       table.NVFP4_B, delta=0.01)
            self.assertEqual(cells[1], "1.40x")
            self.assertEqual(cells[6], "1.40x")


if __name__ == "__main__":
    unittest.main()
