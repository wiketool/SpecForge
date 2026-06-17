import ast
import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[2] / "specforge" / "vlm_sharding.py"
TARGET_MODEL_PATH = (
    Path(__file__).parents[2]
    / "specforge"
    / "modeling"
    / "target"
    / "eagle3_target_model.py"
)
TRAIN_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "train_eagle3.py"
SPEC = importlib.util.spec_from_file_location("vlm_sharding", MODULE_PATH)
vlm_sharding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_sharding)


class TestTrainEagle3VlmSharding(unittest.TestCase):

    def test_get_dp_data_shard_from_tp_matches_tensor_chunk(self):
        tensor = torch.arange(24).view(4, 6)

        for tp_rank in range(4):
            actual = vlm_sharding.get_tp_data_shard(
                tensor, tp_size=4, tp_rank=tp_rank
            )
            expected = tensor.chunk(4, dim=0)[tp_rank]
            self.assertTrue(torch.equal(actual, expected))

    def test_get_dp_data_shard_from_tp_matches_uneven_tensor_chunk(self):
        tensor = torch.arange(30).view(5, 6)

        for tp_rank in range(3):
            actual = vlm_sharding.get_tp_data_shard(
                tensor, tp_size=3, tp_rank=tp_rank
            )
            expected = tensor.chunk(3, dim=0)[tp_rank]
            self.assertTrue(torch.equal(actual, expected))

    def test_get_dp_image_grid_thw_shard_from_tp_handles_mixed_single_sample_shards(
        self,
    ):
        image_grid_thw = [
            torch.tensor([1, 2, 3]),
            None,
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
            None,
        ]

        rank0 = vlm_sharding.get_tp_image_grid_thw_shard(
            image_grid_thw, batch_size=4, tp_size=4, tp_rank=0
        )
        rank1 = vlm_sharding.get_tp_image_grid_thw_shard(
            image_grid_thw, batch_size=4, tp_size=4, tp_rank=1
        )
        rank2 = vlm_sharding.get_tp_image_grid_thw_shard(
            image_grid_thw, batch_size=4, tp_size=4, tp_rank=2
        )

        self.assertTrue(torch.equal(rank0, torch.tensor([[1, 2, 3]])))
        self.assertIsNone(rank1)
        self.assertTrue(torch.equal(rank2, torch.tensor([[4, 5, 6], [7, 8, 9]])))

    def test_get_dp_image_grid_thw_shard_from_tp_flattens_mixed_local_batch(self):
        image_grid_thw = [
            torch.tensor([1, 1, 1]),
            None,
            None,
            torch.tensor([2, 2, 2]),
            torch.tensor([[3, 3, 3], [4, 4, 4]]),
            None,
            None,
            None,
        ]

        rank1 = vlm_sharding.get_tp_image_grid_thw_shard(
            image_grid_thw, batch_size=8, tp_size=4, tp_rank=1
        )
        rank3 = vlm_sharding.get_tp_image_grid_thw_shard(
            image_grid_thw, batch_size=8, tp_size=4, tp_rank=3
        )

        self.assertTrue(torch.equal(rank1, torch.tensor([[2, 2, 2]])))
        self.assertIsNone(rank3)

    def test_vlm_generate_passes_shard_returns_to_extend_vlm(self):
        tree = ast.parse(TARGET_MODEL_PATH.read_text())
        target_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SGLangEagle3TargetModel"
        )
        methods = {
            node.name: node
            for node in target_class.body
            if isinstance(node, ast.FunctionDef)
        }

        extend_vlm_args = {arg.arg for arg in methods["extend_vlm"].args.args}
        self.assertIn("shard_returns", extend_vlm_args)

        extend_vlm_calls = [
            node
            for node in ast.walk(methods["generate_eagle3_data"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extend_vlm"
        ]
        self.assertEqual(len(extend_vlm_calls), 1)
        shard_kw = next(
            (kw for kw in extend_vlm_calls[0].keywords if kw.arg == "shard_returns"),
            None,
        )
        self.assertIsNotNone(shard_kw)
        self.assertIsInstance(shard_kw.value, ast.Name)
        self.assertEqual(shard_kw.value.id, "shard_returns")

    def test_offline_vlm_build_target_model_keeps_processor(self):
        tree = ast.parse(TRAIN_SCRIPT_PATH.read_text())
        build_target_model = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_target_model"
        )
        source = ast.get_source_segment(
            TRAIN_SCRIPT_PATH.read_text(), build_target_model
        )

        self.assertIn("TargetHead.from_pretrained", source)
        self.assertIn("if args.is_vlm:", source)
        self.assertIn("AutoProcessor.from_pretrained", source)
        self.assertIn("return target_head, processor", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
