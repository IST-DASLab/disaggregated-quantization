"""Offline checks for table inputs, coverage, and generated-file exports."""
import tempfile
import json
import unittest
import subprocess
import numpy as np
from pathlib import Path
from unittest.mock import patch

import common
import generate
import make_prefill_table
import make_qadd27b_table
import make_qadd27b_lengths_table
import make_breakdown_table
import make_recorded_table
import make_large_model_table
import make_interop_table


class GeneratorTests(unittest.TestCase):
    def test_interop_uses_complete_final_grids_and_marks_matched_pairs(self):
        generator = make_interop_table
        text = generator.render()
        self.assertIn(r"IQ1\_S & 48.39$^{\star}$ & \textbf{61.54} & 65.02$^{\star}$", text)
        self.assertIn(r"IQ2\_XXS & 66.18 & 67.11 & 66.30 & \textbf{65.90}", text)
        for bench, _, expected in generator.BENCHES:
            cells, keys = generator.grid(bench, expected)
            self.assertEqual(len(cells), 12)
            self.assertEqual(len(keys), expected)
            # A partial cross-run must not silently change the published cohort.
            cells = {key: dict(values) for key, values in cells.items()}
            cells["IQ1_S", "IQ1_M"].pop(next(iter(keys)))
            with patch.object(generator.CG, "load_exported_cells", return_value=cells):
                with self.assertRaises(ValueError):
                    generator.grid(bench, expected)

    def test_cross_grid_export_rejects_missing_cells_and_multiple_passes(self):
        cg = make_interop_table.CG
        with tempfile.TemporaryDirectory() as directory, patch.object(cg, "SCORES", directory):
            path = Path(directory) / "model/mmlu_pro.json"
            path.parent.mkdir()
            tag = cg.tag_for("IQ1_S", "IQ1_S", "reason8k", 980)
            arm = {"ids": ["a", "b"], "scores": [[1], [0]], "repeats": 1}
            path.write_text(json.dumps({"arms": {tag: arm}}))
            cells = cg.load_exported_cells("mmlu_pro", "model", ["IQ1_S"], ["IQ1_S"])
            self.assertEqual(cg.common_keys(cells), {"a", "b"})
            with self.assertRaises(ValueError):
                cg.load_exported_cells("mmlu_pro", "model", ["IQ1_S"], ["RTN", "IQ1_S"])
            arm.update(scores=[[1, 1], [0, 0]], repeats=2)
            path.write_text(json.dumps({"arms": {tag: arm}}))
            with self.assertRaises(ValueError):
                cg.load_exported_cells("mmlu_pro", "model", ["IQ1_S"], ["IQ1_S"])

    def test_cross_grid_intersection_and_exact_test(self):
        cg = make_interop_table.CG
        a = {str(i): False for i in range(6)}
        b = {str(i): True for i in range(6)}
        self.assertEqual(cg.mcnemar(a, b, set(a), bool), (6, 0, 0.03125))
        self.assertEqual(cg.common_keys({"a": {"x": True, "y": False},
                                         "b": {"y": True, "z": True}}), {"y"})
        with self.assertRaises(ValueError):
            cg.common_keys({"a": {"x": True}, "b": {"y": True}})

    def test_generation_lengths_use_final_accuracy_arms(self):
        generator = make_qadd27b_lengths_table
        text = generator.render()
        self.assertIn(r"IQ1\_S & 14579 & 6582 & 10966 & 2393 & 32768 & 29953 & 24.57 & 4.16", text)
        self.assertIn(r"IQ3\_S &", text)
        self.assertIn(r"Q3\_K\_XL &", text)
        data, labels = common.qadd27b_data()
        medians_reduced = 0
        for fmt in labels:
            wo = generator.read_metrics("mmlu_pro", data["mmlu_pro"][fmt, False]["tag"])
            full = generator.read_metrics("mmlu_pro", data["mmlu_pro"][fmt, True]["tag"])
            medians_reduced += full["median"] < wo["median"]
            self.assertGreater(full["p95"], wo["p95"])
            self.assertGreater(full["truncated"], wo["truncated"])
            self.assertGreater(full["mean"], wo["mean"])
        self.assertEqual(medians_reduced, 7)
        for fmt, reduction in (("iq1s", 54.8), ("iq1m", 14.2)):
            wo = generator.read_metrics("mmmu", data["mmmu"][fmt, False]["tag"])
            full = generator.read_metrics("mmmu", data["mmmu"][fmt, True]["tag"])
            self.assertAlmostEqual(100 * (1 - full["mean"] / wo["mean"]), reduction, places=1)

    def test_length_arrays_require_one_complete_valid_pass(self):
        metrics = make_qadd27b_lengths_table.length_metrics
        ids = np.array(["a", "b"])
        # Include zero counts and do not drop the longest/truncated responses.
        self.assertEqual(metrics(ids, np.array([[0], [32768]]), 2, "test")["mean"], 16384)
        for bad_ids, tokens in ((np.array(["a", "a"]), np.ones((2, 1), dtype=int)),
                                (ids, np.ones((2, 2), dtype=int)),
                                (ids, np.array([[-1], [10]])),
                                (ids, np.array([[10], [32769]]))):
            with self.assertRaises(ValueError):
                metrics(bad_ids, tokens, 2, "test")

    def test_generation_lengths_reject_partial_or_invalid_summaries(self):
        generator = make_qadd27b_lengths_table
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(generator, "RESULTS", Path(directory)), \
             patch.object(generator, "SCORES", Path(directory)):
            path = Path(directory) / "mmlu_pro/final/summary.json"
            path.parent.mkdir(parents=True)
            summary = {"n_scored": 12032, "generation": {
                "n": 12032, "truncated": 0, "tokens_p50": 700, "tokens_p95": 1000}}
            np.savez(Path(directory) / "mmlu_pro_lengths.npz",
                     **{"final|ids": np.arange(12032).astype(str),
                        "final|tokens": np.full((12032, 1), 700)})
            path.write_text(json.dumps(summary))
            self.assertEqual(generator.read_metrics("mmlu_pro", "final")["truncated"], 0)
            for key, value in (("n", 12000), ("truncated", -1)):
                changed = {**summary, "generation": {**summary["generation"], key: value}}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    generator.read_metrics("mmlu_pro", "final")
            path.write_text(json.dumps(summary))
            np.savez(Path(directory) / "mmlu_pro_lengths.npz",
                     **{"earlier|ids": np.arange(12032).astype(str),
                        "earlier|tokens": np.full((12032, 1), 700)})
            with self.assertRaises(KeyError):
                generator.read_metrics("mmlu_pro", "final")
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                generator.read_metrics("mmlu_pro", "final")

    def test_qwen_bf16_uses_intended_four_summaries(self):
        generator = make_large_model_table
        for bench, expected in (("mmlu_pro", "84.58"), ("mmmu", "74.78")):
            # Even an absent per-item export must not hide the summary reference.
            with patch.object(generator.CA, "items", return_value={}):
                cells, incomplete = generator.row_for("qwen3.8-27b", bench)
            self.assertEqual(cells["BF16"], expected)
            self.assertFalse(incomplete)

    def test_qwen_bf16_summary_selection_and_validation(self):
        generator = make_large_model_table
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(generator.CA, "RESULTS_ROOT", directory):
            root = Path(directory) / "qwen3.8-27b/mmlu_pro"
            for tag in (*generator.BASELINE_SUMMARY_TAGS["qwen3.8-27b", "mmlu_pro"],
                        "bf16-r5", "bf16-b200", "bf16-pd"):
                path = root / tag / "summary.json"
                path.parent.mkdir(parents=True)
                accuracy = 0.5 if tag in ("bf16-r1", "bf16-r2", "bf16-r3", "bf16-r4") else 1.0
                path.write_text(json.dumps({"accuracy": accuracy, "n_scored": 12032}))
            self.assertEqual(generator._baseline_summary_cell("qwen3.8-27b", "mmlu_pro"),
                             (50.0, 12032, 4))
            path = root / "bf16-r4/summary.json"
            for summary in ({"accuracy": 0.5, "n_scored": 12031},
                            {"accuracy": 1.5, "n_scored": 12032}):
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    generator._baseline_summary_cell("qwen3.8-27b", "mmlu_pro")
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                generator._baseline_summary_cell("qwen3.8-27b", "mmlu_pro")

    def test_all_manuscript_tables_have_generators(self):
        import re
        tex = (common.ROOT / "latex/main.tex").read_text()
        labels = set(re.findall(r"\\label\{tab:([^}]+)\}", tex))
        self.assertEqual(labels, set(generate.GENERATORS))
        for label in labels:
            self.assertEqual(tex.count(rf"\input{{tables/{label}.tex}}"), 1)
            output = (common.NOTEBOOKS / "tables" / f"{label}.tex").read_text()
            self.assertIn(r"\begin{tabular}", output)
            self.assertNotIn(r"\caption", output)
            self.assertNotIn(r"\label", output)

    def test_normalization_preserves_escaped_percent(self):
        self.assertEqual(generate.normalized("a & 1 % note\n"), "a&1")
        self.assertNotEqual(generate.normalized(r"\%"), "")

    def test_duplicate_csv_measurement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("model,value\nx,1\nx,2\n")
            with self.assertRaises(ValueError):
                common.csv_index(path, ("model",))

    def test_write_exports_tex_without_touching_manuscript(self):
        new = r"\begin{tabular}{lr}a & 2 \\\end{tabular}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.tex"
            path.write_text("untouched")
            output = Path(directory) / "tables"
            with patch("sys.argv", ["generate.py", "hyper", "--write", "--output-dir", str(output)]), \
                 patch.object(generate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, new, "")):
                generate.main()
            self.assertEqual(path.read_text(), "untouched")
            self.assertEqual(generate.normalized((output / "hyper.tex").read_text()), generate.normalized(new))

    def test_failed_generation_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hyper.tex"
            path.write_text("untouched")
            with patch("sys.argv", ["generate.py", "hyper", "--write", "--output-dir", directory]), \
                 patch.object(generate.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                with self.assertRaises(SystemExit):
                    generate.main()
            self.assertEqual(path.read_text(), "untouched")

    def test_check_rejects_missing_output_without_writing(self):
        body = r"\begin{tabular}{lr}a & 1 \\\end{tabular}"
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["generate.py", "hyper", "--check", "--output-dir", directory]), \
                 patch.object(generate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, body, "")):
                with self.assertRaises(SystemExit):
                    generate.main()
            self.assertFalse((Path(directory) / "hyper.tex").exists())

    def test_qadd_includes_higher_formats_and_final_steps(self):
        data, labels = common.qadd27b_data()
        self.assertEqual(len(labels), 8)
        text = make_qadd27b_table.render()
        self.assertIn(r"IQ3\_S &", text)
        self.assertIn(r"Q3\_K\_XL &", text)
        self.assertIn("+32.50", text)
        for bench in data:
            for fmt in labels:
                self.assertTrue(data[bench][fmt, True]["tag"].startswith("reason8k980-"))

    def test_prefill_requires_zero_ssd(self):
        data = common.prefill_data()
        del data["Qwen/Qwen3-8B", "nvfp4", "zero-ssd", "16384"]
        with patch.object(make_prefill_table, "prefill_data", return_value=data):
            with self.assertRaises(KeyError):
                make_prefill_table.render()

    def test_other_generators_emit_tabulars(self):
        for text in [make_prefill_table.render(), make_breakdown_table.render(),
                     *[make_recorded_table.render(n) for n in ("hyper", "parallel", "models", "attn-backend")]]:
            self.assertTrue(text.startswith(r"\begin{tabular}"))
            self.assertTrue(text.endswith(r"\end{tabular}"))

    def test_ptq_requires_four_repeats_on_both_benchmarks(self):
        self.assertEqual(make_large_model_table.REPEATS, 4)
        for bench, k in (("mmlu_pro", 4), ("mmmu", 4), ("mmlu_pro", 1), ("mmmu", 1)):
            with patch.object(make_large_model_table, "_baseline", return_value="bf16"), \
                 patch.object(make_large_model_table, "_first_present", return_value=None), \
                 patch.object(make_large_model_table.CA, "items", return_value={"x": [1]}), \
                 patch.object(make_large_model_table.CA, "expected_items", return_value=100), \
                 patch.object(make_large_model_table, "_cell", side_effect=lambda b, tag: (50., 100, k) if tag else None):
                _, incomplete = make_large_model_table.row_for("model", bench)
                self.assertEqual(incomplete, k != 4)


if __name__ == "__main__":
    unittest.main()
