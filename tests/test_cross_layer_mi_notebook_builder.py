import unittest

from scripts import build_cross_layer_mi_colab_notebook


def notebook_source() -> str:
    notebook = build_cross_layer_mi_colab_notebook.build_notebook()
    chunks = []
    for cell in notebook["cells"]:
        chunks.extend(cell["source"])
    return "".join(chunks)


class CrossLayerMiNotebookBuilderTests(unittest.TestCase):
    def test_notebook_uses_persistent_results_dir_for_disconnect_safety(self):
        source = notebook_source()

        self.assertIn("RESULTS_DIR", source)
        self.assertIn("SUB1QUANT_SAVE_DIR", source)
        self.assertIn("--progress-output", source)

    def test_notebook_runs_mi_biased_mixed_budget_scan(self):
        source = notebook_source()

        self.assertIn("scripts/scan_mixed_budget.py", source)
        self.assertIn("--mi-report", source)
        self.assertIn("--mi-prior", source)
        self.assertIn("cross_layer_mi_colab_delta.json", source)


if __name__ == "__main__":
    unittest.main()
