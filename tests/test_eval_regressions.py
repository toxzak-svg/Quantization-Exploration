import ast
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import eval_quantized


class FakeModule:
    def __init__(self, shape):
        self.weight = torch.zeros(shape)


class FakeModel:
    def __init__(self):
        self.down = FakeModule((2, 2))
        self.up = FakeModule((2, 2))

    def named_modules(self):
        return [
            ("", self),
            ("model.layers.0.down_proj", self.down),
            ("model.layers.0.up_proj", self.up),
        ]


class FakeSharedKvModel:
    def __init__(self):
        self.q = FakeModule((2, 2))
        self.o = FakeModule((2, 2))

    def named_modules(self):
        return [
            ("", self),
            ("model.layers.15.self_attn.q_proj", self.q),
            ("model.layers.15.self_attn.o_proj", self.o),
        ]


class EvalRegressionTests(unittest.TestCase):
    def test_apply_quantized_weights_uses_exact_weight_key_not_shape_order(self):
        model = FakeModel()
        quantized = {
            0: {
                "key": "model.layers.0.up_proj.weight",
                "orig_shape": [2, 2],
            }
        }

        def reconstruct(_entry, _device):
            return torch.ones((2, 2))

        stats = eval_quantized.apply_quantized_weights(
            model,
            quantized,
            device="cpu",
            reconstruct_fn=reconstruct,
        )

        self.assertEqual(stats["replaced"], 1)
        self.assertTrue(torch.equal(model.up.weight, torch.ones((2, 2))))
        self.assertTrue(torch.equal(model.down.weight, torch.zeros((2, 2))))

    def test_apply_quantized_weights_skips_expected_shared_kv_entries(self):
        model = FakeSharedKvModel()
        quantized = {
            0: {
                "key": "model.layers.15.self_attn.k_proj.weight",
                "orig_shape": [2, 2],
            }
        }

        stats = eval_quantized.apply_quantized_weights(
            model,
            quantized,
            device="cpu",
            reconstruct_fn=lambda _entry, _device: torch.ones((2, 2)),
        )

        self.assertEqual(stats["replaced"], 0)
        self.assertEqual(
            stats["skipped"],
            ["model.layers.15.self_attn.k_proj.weight"],
        )

    def test_resolve_quantized_weight_key_uses_checkpoint_index_for_legacy_entries(self):
        key = eval_quantized.resolve_quantized_weight_key(
            1,
            {"original_shape": [2, 2]},
            [
                ("model.layers.0.down_proj.weight", (2, 2)),
                ("model.layers.0.up_proj.weight", (2, 2)),
            ],
        )

        self.assertEqual(key, "model.layers.0.up_proj.weight")

    def test_load_model_weight_keys_prefers_language_model_weights(self):
        header = {
            "model.embed_audio.embedding_projection.weight": {
                "dtype": "F16",
                "shape": [1, 1],
                "data_offsets": [0, 2],
            },
            "model.language_model.layers.0.down_proj.weight": {
                "dtype": "F16",
                "shape": [2, 2],
                "data_offsets": [2, 10],
            },
            "model.language_model.layers.0.up_proj.weight": {
                "dtype": "F16",
                "shape": [2, 2],
                "data_offsets": [10, 18],
            },
        }
        header_bytes = json.dumps(header).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            safetensor = model_dir / "model.safetensors"
            safetensor.write_bytes(
                struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\0" * 18)
            )

            keys = eval_quantized.load_model_weight_keys(model_dir)

        self.assertEqual(
            keys,
            [
                ("model.language_model.layers.0.down_proj.weight", (2, 2)),
                ("model.language_model.layers.0.up_proj.weight", (2, 2)),
            ],
        )

    def test_perplexity_script_uses_real_perplexity_evaluator(self):
        source = Path("test_perplexity.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertNotIn("calculate_perplexity_simple", function_names)
        self.assertIn("eval_perplexity", source)


if __name__ == "__main__":
    unittest.main()
