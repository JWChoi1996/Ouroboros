import tempfile
import unittest
from pathlib import Path

import numpy as np


class FakeTensor:
    def __init__(self, array, device=None):
        self.array = np.asarray(array)
        self.shape = self.array.shape
        self.dtype = "float32"
        self.device = device

    def numel(self):
        return self.array.size

    def element_size(self):
        return self.array.dtype.itemsize


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        return FakeTensor(np.asarray(value, dtype=np.float32), device=device)


class FakeModel:
    def __init__(self):
        self.calls = []

    def forward(self, frame, *, timer=None):
        self.calls.append(("forward", frame, timer))
        return ("dense",)

    def _cached(self, name, frame, anchor_features, dirtiness_map, timer):
        self.calls.append(
            (name, frame, anchor_features, dirtiness_map, timer)
        )
        return (name, {"nested": [FakeTensor(np.zeros(3, np.float32))]})

    def forward_contexted(self, frame, *, anchor_features, dirtiness_map, timer=None):
        return self._cached(
            "forward_contexted", frame, anchor_features, dirtiness_map, timer
        )

    def forward_eventful(self, frame, *, anchor_features, dirtiness_map, timer=None):
        return self._cached(
            "forward_eventful", frame, anchor_features, dirtiness_map, timer
        )

    def forward_maskvd(self, frame, *, anchor_features, dirtiness_map, timer=None):
        return self._cached(
            "forward_maskvd", frame, anchor_features, dirtiness_map, timer
        )

    def forward_stgt(self, frame, *, anchor_features, dirtiness_map, timer=None):
        return self._cached(
            "forward_stgt", frame, anchor_features, dirtiness_map, timer
        )


class M1VitDetBAdapterTests(unittest.TestCase):
    def test_b1_lineage_is_native_but_measurement_provenance_is_adapted(self):
        from on_device.motivation_m1.vitdet_b_adapter import (
            MEASUREMENT_FIDELITY,
            METHOD_LINEAGE,
            PROVENANCE,
        )

        self.assertEqual(PROVENANCE["B1-MV"], "ADAPTED")
        self.assertEqual(METHOD_LINEAGE["B1-MV"], "NATIVE_MV_LINEAGE")
        self.assertEqual(
            MEASUREMENT_FIDELITY["B1-MV"], "ADAPTED_DECODER_MV"
        )

    def test_exact_mask_conversion_is_explicit_float_shape_and_device(self):
        from on_device.motivation_m1.vitdet_b_adapter import exact_forced_mask

        tensor = exact_forced_mask(
            np.ones((64, 64), dtype=bool),
            "cuda:7",
            torch_module=FakeTorch,
        )

        self.assertEqual(tensor.shape, (1, 64, 64, 1))
        self.assertEqual(tensor.dtype, "float32")
        self.assertEqual(tensor.device, "cuda:7")

    def test_dispatch_calls_exact_method_and_updates_cache(self):
        from on_device.motivation_m1.vitdet_b_adapter import VitDetBAdapter

        expected = {
            "B0": "forward",
            "B1-MV": "forward_contexted",
            "B2": "forward_eventful",
            "B3": "forward_maskvd",
            "B4": "forward_stgt",
        }
        for method, method_name in expected.items():
            with self.subTest(method=method):
                model = FakeModel()
                adapter = VitDetBAdapter(
                    model=model, device="fake:0", torch_module=FakeTorch
                )
                output = adapter.forward(
                    method,
                    frame="frame",
                    forced_mask=np.ones((64, 64), dtype=bool),
                    timer="timer",
                )
                self.assertEqual(model.calls[0][0], method_name)
                if method == "B0":
                    self.assertEqual(output, ("dense",))
                    self.assertEqual(adapter.cache_bytes(), 0)
                else:
                    self.assertEqual(output, method_name)
                    self.assertEqual(adapter.cache_bytes(), 12)

    def test_reset_cache_removes_nested_cache(self):
        from on_device.motivation_m1.vitdet_b_adapter import VitDetBAdapter

        adapter = VitDetBAdapter(
            model=FakeModel(), device="fake:0", torch_module=FakeTorch
        )
        adapter.forward(
            "B2", "frame", np.ones((64, 64), dtype=bool), timer=None
        )
        adapter.reset_cache()

        self.assertEqual(adapter.cache_bytes(), 0)

    def test_detections_with_different_counts_are_not_equivalent(self):
        from on_device.motivation_m1.vitdet_b_adapter import (
            detections_equivalent,
        )

        six_detections = (
            np.zeros((6, 4), dtype=np.float32),
            np.zeros(6, dtype=np.int64),
            np.zeros(6, dtype=np.float32),
        )
        four_detections = (
            np.zeros((4, 4), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.float32),
        )

        self.assertFalse(
            detections_equivalent(six_detections, four_detections)
        )

    def test_weight_coverage_counts_only_shape_matched_keys_and_bytes(self):
        from on_device.motivation_m1.vitdet_b_adapter import weight_coverage

        model_state = {
            "base_model.backbone.a": FakeTensor(np.zeros(2, np.float32)),
            "base_model.roi_heads.b": FakeTensor(np.zeros(4, np.float32)),
            "base_model.roi_heads.c": FakeTensor(np.zeros(3, np.float32)),
        }
        checkpoint = {
            "backbone.a": FakeTensor(np.ones(2, np.float32)),
            "roi_heads.b": FakeTensor(np.ones(5, np.float32)),
            "extra": FakeTensor(np.ones(1, np.float32)),
        }

        report = weight_coverage(model_state, checkpoint)

        self.assertEqual(report["matched_key_count"], 1)
        self.assertAlmostEqual(report["matched_parameter_byte_ratio"], 8 / 36)
        self.assertEqual(
            report["missing_keys"],
            ["base_model.roi_heads.b", "base_model.roi_heads.c"],
        )
        self.assertEqual(report["unexpected_keys"], ["base_model.extra"])
        self.assertEqual(report["backbone_coverage"]["matched_keys"], 1)
        self.assertEqual(report["head_coverage"]["matched_keys"], 0)

    def test_checkpoint_metadata_includes_sha256(self):
        from on_device.motivation_m1.vitdet_b_adapter import checkpoint_sha256

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pkl"
            path.write_bytes(b"checkpoint")
            self.assertEqual(
                checkpoint_sha256(path),
                "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
            )

    def test_checkpoint_report_requires_exact_full_model_coverage(self):
        from on_device.motivation_m1.vitdet_b_adapter import (
            validate_checkpoint_report,
        )

        exact = {
            "matched_parameter_byte_ratio": 1.0,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatched_keys": [],
            "load_missing_keys": [],
            "load_unexpected_keys": [],
        }
        validate_checkpoint_report(exact)

        invalid = (
            {**exact, "matched_parameter_byte_ratio": 0.0},
            {**exact, "matched_parameter_byte_ratio": 0.99},
            {**exact, "missing_keys": ["base_model.missing"]},
            {**exact, "unexpected_keys": ["base_model.extra"]},
            {**exact, "shape_mismatched_keys": ["base_model.wrong_shape"]},
            {**exact, "load_missing_keys": ["missing_after_load"]},
            {**exact, "load_unexpected_keys": ["unexpected_after_load"]},
        )
        for report in invalid:
            with self.subTest(report=report), self.assertRaisesRegex(
                ValueError, "exact full model coverage"
            ):
                validate_checkpoint_report(report)


if __name__ == "__main__":
    unittest.main()
