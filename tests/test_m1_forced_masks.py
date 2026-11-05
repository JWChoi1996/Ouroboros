import unittest


ALLOWED_DIRTY_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05)
GRID_SHAPE = (64, 64)


class M1ForcedMaskTests(unittest.TestCase):
    @staticmethod
    def _make_mask(ratio, seed=2026):
        from on_device.motivation_m1.forced_masks import make_forced_mask

        return make_forced_mask(GRID_SHAPE, ratio, seed=seed)

    def test_supported_ratios_have_exact_k_active_cells(self):
        cell_count = GRID_SHAPE[0] * GRID_SHAPE[1]

        for ratio in ALLOWED_DIRTY_RATIOS:
            with self.subTest(ratio=ratio):
                mask = self._make_mask(ratio)
                self.assertEqual(mask.shape, GRID_SHAPE)
                self.assertEqual(mask.dtype, bool)
                self.assertEqual(sum(mask.flat), round(cell_count * ratio))

    def test_same_seed_and_ratio_produce_identical_mask(self):
        first = self._make_mask(0.1, seed=17)
        second = self._make_mask(0.1, seed=17)

        self.assertEqual(first.tolist(), second.tolist())

    def test_different_seeds_change_nontrivial_mask_without_changing_count(self):
        first = self._make_mask(0.25, seed=17)
        second = self._make_mask(0.25, seed=18)

        self.assertNotEqual(first.tolist(), second.tolist())
        self.assertEqual(sum(first.flat), sum(second.flat))

    def test_ratio_outside_unit_interval_is_rejected(self):
        for ratio in (-0.01, 0.0, 1.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self._make_mask(ratio)

    def test_ratio_not_in_protocol_sweep_is_rejected(self):
        with self.assertRaises(ValueError):
            self._make_mask(0.2)


if __name__ == "__main__":
    unittest.main()
