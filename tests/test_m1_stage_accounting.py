import unittest


class FakeTensor:
    def __init__(self, elements, bytes_per_element):
        self._elements = elements
        self._bytes_per_element = bytes_per_element

    def numel(self):
        return self._elements

    def element_size(self):
        return self._bytes_per_element


class M1StageAccountingTests(unittest.TestCase):
    @staticmethod
    def _timings():
        from on_device.motivation_m1.accounting import StageTimings

        return StageTimings(
            encode_ms=2.0,
            decode_ms=3.0,
            rgb_materialization_ms=5.0,
            mv_export_ms=7.0,
            residual_export_ms=1000.0,
            affine_ms=11.0,
            warp_ms=13.0,
            diff_ms=17.0,
            mask_ms=19.0,
            model_preprocess_ms=23.0,
            model_forward_ms=29.0,
        )

    def test_rgb_baselines_charge_decode_materialization_and_model_only_to_e2e(self):
        from on_device.motivation_m1.accounting import account_stages

        expected_e2e = 3.0 + 5.0 + 23.0 + 29.0
        for method in ("B0", "B2", "B3", "B4"):
            with self.subTest(method=method):
                totals = account_stages(method, self._timings())
                self.assertEqual(totals.decode_to_output_e2e_ms, expected_e2e)
                self.assertEqual(
                    totals.source_to_output_total_ms,
                    2.0 + expected_e2e,
                )

    def test_rgb_baselines_exclude_encode_and_metadata_exports_from_e2e(self):
        from on_device.motivation_m1.accounting import account_stages

        totals = account_stages("B2", self._timings())

        self.assertEqual(totals.decode_to_output_e2e_ms, 60.0)
        self.assertNotEqual(totals.decode_to_output_e2e_ms, 62.0)
        self.assertLess(totals.decode_to_output_e2e_ms, 1000.0)

    def test_b1_mv_native_total_uses_raw_rgb_and_charges_its_own_pipeline(self):
        from on_device.motivation_m1.accounting import account_stages

        totals = account_stages("B1-MV", self._timings())

        self.assertIsNone(totals.decode_to_output_e2e_ms)
        self.assertEqual(
            totals.source_to_output_total_ms,
            2.0 + 7.0 + 11.0 + 13.0 + 17.0 + 19.0 + 23.0 + 29.0,
        )

    def test_b1_mv_excludes_decode_rgb_materialization_and_residual_export(self):
        from on_device.motivation_m1.accounting import account_stages

        totals = account_stages("B1-MV", self._timings())

        self.assertEqual(totals.source_to_output_total_ms, 121.0)
        self.assertLess(totals.source_to_output_total_ms, 1000.0)

    def test_residual_export_is_never_charged_to_any_method(self):
        from on_device.motivation_m1.accounting import account_stages

        timings = self._timings()
        for method in ("B0", "B1-MV", "B2", "B3", "B4"):
            with self.subTest(method=method):
                total = account_stages(method, timings).source_to_output_total_ms
                self.assertLess(total, timings.residual_export_ms)

    def test_nested_tensor_mapping_bytes_counts_every_tensor_leaf(self):
        from on_device.motivation_m1.accounting import nested_tensor_bytes

        cache = {
            "block_0": {
                "qkv": FakeTensor(12, 2),
                "output": FakeTensor(5, 4),
            },
            "block_1": {"attention": {"lse": FakeTensor(3, 8)}},
        }

        self.assertEqual(nested_tensor_bytes(cache), 12 * 2 + 5 * 4 + 3 * 8)

    def test_nested_tensor_mapping_bytes_ignores_non_tensor_metadata(self):
        from on_device.motivation_m1.accounting import nested_tensor_bytes

        cache = {
            "block_0": FakeTensor(10, 2),
            "metadata": {"age": 4, "name": "reference"},
            "empty": None,
        }

        self.assertEqual(nested_tensor_bytes(cache), 20)


if __name__ == "__main__":
    unittest.main()
