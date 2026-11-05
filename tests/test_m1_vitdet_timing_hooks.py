import ast
from contextlib import contextmanager
from functools import wraps
import inspect
from pathlib import Path
import textwrap
import unittest


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "on_device"
    / "ipconv"
    / "models"
    / "ViTDet"
    / "maskedrcnn_vit_fpn.py"
)
TIMED_METHODS = {
    "forward",
    "forward_contexted",
    "forward_eventful",
    "forward_maskvd",
    "forward_stgt",
}


class M1ViTDetTimingHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODEL_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.methods = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_vitdet_methods_accept_an_optional_timer(self):
        for method_name in TIMED_METHODS:
            with self.subTest(method=method_name):
                method = self.methods[method_name]
                arguments = method.args.args + method.args.kwonlyargs
                defaults = method.args.defaults + method.args.kw_defaults
                timer_index = next(
                    index
                    for index, argument in enumerate(arguments)
                    if argument.arg == "timer"
                )
                default_offset = len(arguments) - len(defaults)
                default = defaults[timer_index - default_offset]
                self.assertIsInstance(default, ast.Constant)
                self.assertIsNone(default.value)

    def test_model_source_exposes_required_stage_boundaries(self):
        for stage_name in (
            "model_preprocess",
            "model_total",
            "patch_embedding",
            "block_qkv",
            "block_attention",
            "block_mlp",
            "block_cache",
            "fpn",
            "detector_heads",
        ):
            with self.subTest(stage=stage_name):
                self.assertIn(f'"{stage_name}"', self.source)

    def test_dense_forward_does_not_claim_backbone_is_patch_embedding(self):
        dense_forward = self.methods["forward"]
        stage_names = {
            node.value
            for node in ast.walk(dense_forward)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("dense_backbone", stage_names)
        self.assertNotIn("patch_embedding", stage_names)

    def test_dense_detector_calls_are_owned_by_base_model(self):
        dense_source = ast.get_source_segment(
            self.source, self.methods["forward"]
        )

        for operation in ("proposal_generator", "roi_heads", "_postprocess"):
            with self.subTest(operation=operation):
                self.assertIn(f"self.base_model.{operation}(", dense_source)
                self.assertNotIn(f"lambda: self.{operation}(", dense_source)

    def test_timing_helper_closes_stage_when_operation_raises(self):
        helper = self.methods["_timed_call"]
        helper_source = ast.get_source_segment(self.source, helper)
        namespace = {}
        exec(textwrap.dedent(helper_source), namespace)

        exits = []

        class Stage:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                exits.append(exc_type)

        class Timer:
            def stage(self, name, cuda):
                return Stage()

        def fail():
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            namespace["_timed_call"](Timer(), "stage", fail)
        self.assertEqual(exits, [RuntimeError])

    def test_model_contains_no_manual_context_protocol_calls(self):
        self.assertNotIn(".__enter__(", self.source)
        self.assertNotIn(".__exit__(", self.source)

    def test_each_contexted_path_charges_multiple_cache_operations(self):
        for method_name in TIMED_METHODS - {"forward"}:
            method = self.methods[method_name]
            method_source = ast.get_source_segment(self.source, method)
            with self.subTest(method=method_name):
                self.assertGreaterEqual(method_source.count('"block_cache"'), 2)

            timed_cache_nodes = {
                id(descendant)
                for call in ast.walk(method)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_timed_call"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and call.args[1].value == "block_cache"
                for descendant in ast.walk(call)
            }
            anchor_reads = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "anchor_features"
            ]
            direct_cache_writes = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"anchor_features", "new_cache_feature"}
            ]
            with self.subTest(method=method_name, operation="cache reads"):
                self.assertTrue(anchor_reads)
                self.assertTrue(
                    all(id(read) in timed_cache_nodes for read in anchor_reads)
                )
            with self.subTest(method=method_name, operation="cache writes"):
                self.assertEqual(direct_cache_writes, [])

    def test_no_timer_helper_preserves_result_and_calls_operation_once(self):
        helper = self.methods["_timed_call"]
        helper_source = ast.get_source_segment(self.source, helper)
        namespace = {}
        exec(textwrap.dedent(helper_source), namespace)
        calls = []

        result = namespace["_timed_call"](
            None, "stage", lambda: calls.append("called") or object()
        )

        self.assertEqual(calls, ["called"])
        self.assertIsNotNone(result)

    def test_every_forward_has_whole_method_model_total_decorator(self):
        for method_name in TIMED_METHODS:
            decorators = self.methods[method_name].decorator_list
            decorator_names = {
                decorator.id
                for decorator in decorators
                if isinstance(decorator, ast.Name)
            }
            with self.subTest(method=method_name):
                self.assertIn("_timed_model_total", decorator_names)

    def test_preprocess_span_contains_complete_image_preparation(self):
        for method_name in TIMED_METHODS:
            method = self.methods[method_name]
            preprocess_spans = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.With)
                and any(
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == "_timed_stage"
                    and len(item.context_expr.args) >= 2
                    and isinstance(item.context_expr.args[1], ast.Constant)
                    and item.context_expr.args[1].value == "model_preprocess"
                    for item in node.items
                )
            ]
            with self.subTest(method=method_name):
                self.assertEqual(len(preprocess_spans), 1)
                span_source = ast.get_source_segment(
                    self.source, preprocess_spans[0]
                )
                self.assertIn("torch.tensor", span_source)
                self.assertIn("_move_to_current_device", span_source)
                self.assertIn("pixel_mean", span_source)
                self.assertIn("pixel_std", span_source)
                self.assertIn("ImageList.from_tensors", span_source)

    def test_every_forward_uses_backbone_square_padding_constraints(self):
        for method_name in TIMED_METHODS:
            method_source = ast.get_source_segment(
                self.source, self.methods[method_name]
            )
            with self.subTest(method=method_name):
                self.assertIn(
                    "padding_constraints=self.base_model.backbone.padding_constraints",
                    method_source,
                )
                self.assertNotIn('"padding_constraints": image_ndarray.shape[0]', method_source)

    def test_vid_shape_pads_to_sixty_four_by_sixty_four_tokens(self):
        import torch
        from on_device.ipconv.models.ViTDet.structures.image_list import ImageList

        images = ImageList.from_tensors(
            [torch.zeros((3, 480, 640))],
            size_divisibility=32,
            padding_constraints={"square_size": 1024},
        )

        self.assertEqual(tuple(images.tensor.shape), (1, 3, 1024, 1024))
        self.assertEqual(
            (images.tensor.shape[-2] // 16, images.tensor.shape[-1] // 16),
            (64, 64),
        )

    def test_attention_compute_is_owned_by_attention_stage(self):
        attention_operations = {"matmul", "add_decomposed_rel_pos", "softmax", "proj"}
        for method_name in TIMED_METHODS - {"forward"}:
            method = self.methods[method_name]
            timed_attention_nodes = {
                id(descendant)
                for call in ast.walk(method)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_timed_call"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and call.args[1].value == "block_attention"
                for descendant in ast.walk(call)
            }
            operations = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in attention_operations
            ]
            with self.subTest(method=method_name):
                self.assertTrue(operations)
                self.assertTrue(
                    all(id(operation) in timed_attention_nodes for operation in operations)
                )

    def test_temporary_attention_reassembly_is_not_cache_timing(self):
        for method_name in TIMED_METHODS - {"forward"}:
            method = self.methods[method_name]
            cache_calls = [
                call
                for call in ast.walk(method)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_timed_call"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and call.args[1].value == "block_cache"
            ]
            temporary_scatter_targets = []
            for cache_call in cache_calls:
                for descendant in ast.walk(cache_call):
                    if (
                        isinstance(descendant, ast.Call)
                        and isinstance(descendant.func, ast.Name)
                        and descendant.func.id == "_scatter"
                        and descendant.args
                        and isinstance(descendant.args[0], ast.Name)
                        and descendant.args[0].id in {"attn", "x_attn", "qkv_flat"}
                    ):
                        temporary_scatter_targets.append(descendant.args[0].id)
            with self.subTest(method=method_name):
                self.assertEqual(temporary_scatter_targets, [])

    def test_large_temporary_allocations_are_owned_by_served_stage(self):
        for method_name in TIMED_METHODS - {"forward"}:
            method = self.methods[method_name]
            timed_allocations = {}
            for call in ast.walk(method):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_timed_call"
                    and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant)
                ):
                    for descendant in ast.walk(call):
                        timed_allocations[id(descendant)] = call.args[1].value

            allocations = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr in {"zeros", "zeros_like"}
            ]
            with self.subTest(method=method_name):
                self.assertTrue(allocations)
                self.assertTrue(
                    all(
                        timed_allocations.get(id(allocation))
                        in {"block_qkv", "block_attention", "block_cache"}
                        for allocation in allocations
                    )
                )

            for allocation in allocations:
                parent_assignment = next(
                    (
                        node
                        for node in ast.walk(method)
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                        and allocation in list(ast.walk(node))
                    ),
                    None,
                )
                if (
                    isinstance(parent_assignment, ast.Assign)
                    and len(parent_assignment.targets) == 1
                    and isinstance(parent_assignment.targets[0], ast.Name)
                ):
                    target = parent_assignment.targets[0].id
                    expected = {
                        "qkv_flat": "block_qkv",
                        "attn": "block_attention",
                        "x_attn": "block_attention",
                    }.get(target)
                    if expected is not None:
                        with self.subTest(method=method_name, target=target):
                            self.assertEqual(
                                timed_allocations.get(id(allocation)), expected
                            )

    def test_dead_qkv_cached_zeros_like_allocation_is_absent(self):
        contexted = self.methods["forward_contexted"]
        dead_allocations = [
            assignment
            for assignment in ast.walk(contexted)
            if isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == "qkv_cached"
            and any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "zeros_like"
                for node in ast.walk(assignment.value)
            )
        ]

        self.assertEqual(dead_allocations, [])

    def test_dense_legacy_positional_args_are_never_treated_as_timer(self):
        helper = self.methods["_timed_model_total"]
        helper_source = ast.get_source_segment(self.source, helper)
        observed_timers = []

        @contextmanager
        def fake_timed_stage(timer, name):
            observed_timers.append(timer)
            yield

        namespace = {
            "inspect": inspect,
            "wraps": wraps,
            "_timed_stage": fake_timed_stage,
        }
        exec(textwrap.dedent(helper_source), namespace)

        @namespace["_timed_model_total"]
        def dense(self, image, *args, timer=None, **kwargs):
            return args

        legacy_args = tuple(object() for _ in range(10))
        self.assertEqual(dense(None, None, *legacy_args), legacy_args)
        self.assertEqual(observed_timers, [None])


if __name__ == "__main__":
    unittest.main()
