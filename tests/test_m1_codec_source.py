import importlib
import sys
import unittest

import numpy as np


class M1CodecSourceTests(unittest.TestCase):
    def test_import_does_not_load_native_codec(self):
        sys.modules.pop("on_device.motivation_m1.codec_source", None)
        sys.modules.pop("on_device.codecs_h264.h264_pr_raw", None)

        importlib.import_module("on_device.motivation_m1.codec_source")

        self.assertNotIn("on_device.codecs_h264.h264_pr_raw", sys.modules)

    def test_method_codec_flags_never_enable_residuals(self):
        from on_device.motivation_m1.codec_source import codec_flags

        for method in ("B0", "B2", "B3", "B4"):
            with self.subTest(method=method):
                self.assertEqual(
                    codec_flags(method),
                    {"export_mvs": False, "export_residual": False},
                )
        self.assertEqual(
            codec_flags("B1-MV"),
            {"export_mvs": True, "export_residual": False},
        )

    def test_source_selection_uses_raw_only_for_mv_method(self):
        from on_device.motivation_m1.codec_source import select_model_frame

        raw = object()
        reconstructed = object()
        self.assertIs(select_model_frame("B1-MV", raw, reconstructed), raw)
        for method in ("B0", "B2", "B3", "B4"):
            self.assertIs(
                select_model_frame(method, raw, reconstructed), reconstructed
            )

    def test_native_mv_rows_are_adapted_to_rigid_contract(self):
        from on_device.motivation_m1.codec_source import adapt_native_mvs

        native = np.array(
            [
                [10, 20, 8, 4, 12, -6, 4, -1, 0],
                [3, 7, 2, 6, -2, 8, 2, 1, 1],
            ],
            dtype=np.int32,
        )

        adapted = adapt_native_mvs(native)

        np.testing.assert_array_equal(
            adapted,
            np.array([[14, 22, 12, -6, 4], [4, 10, -2, 8, 2]], dtype=np.float32),
        )

    def test_mv_adapter_rejects_non_native_shape(self):
        from on_device.motivation_m1.codec_source import adapt_native_mvs

        with self.assertRaisesRegex(ValueError, "Nx9"):
            adapt_native_mvs(np.zeros((2, 5)))


if __name__ == "__main__":
    unittest.main()
