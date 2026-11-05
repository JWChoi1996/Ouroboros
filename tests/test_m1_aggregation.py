import unittest


class M1AggregationTests(unittest.TestCase):
    @staticmethod
    def _records():
        return [
            {
                "frame_index": index,
                "decode_to_output_e2e_ms": float(index + 1),
                "source_to_output_total_ms": float((index + 1) * 10),
            }
            for index in range(10)
        ]

    def test_percentiles_use_linear_interpolation(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        result = aggregate_percentiles(
            self._records(),
            metrics=("decode_to_output_e2e_ms",),
            warmup_frames=0,
            percentiles=(50, 95),
        )

        self.assertEqual(result["decode_to_output_e2e_ms"]["p50"], 5.5)
        self.assertAlmostEqual(result["decode_to_output_e2e_ms"]["p95"], 9.55)

    def test_warmup_frames_are_excluded_before_aggregation(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        result = aggregate_percentiles(
            self._records(),
            metrics=("decode_to_output_e2e_ms",),
            warmup_frames=2,
            percentiles=(50, 95),
        )

        self.assertEqual(result["sample_count"], 8)
        self.assertEqual(result["excluded_warmup_frames"], 2)
        self.assertEqual(result["decode_to_output_e2e_ms"]["p50"], 6.5)
        self.assertAlmostEqual(result["decode_to_output_e2e_ms"]["p95"], 9.65)

    def test_each_requested_metric_is_aggregated_over_the_same_measured_frames(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        result = aggregate_percentiles(
            self._records(),
            metrics=(
                "decode_to_output_e2e_ms",
                "source_to_output_total_ms",
            ),
            warmup_frames=2,
            percentiles=(50,),
        )

        self.assertEqual(result["decode_to_output_e2e_ms"]["p50"], 6.5)
        self.assertEqual(result["source_to_output_total_ms"]["p50"], 65.0)

    def test_aggregation_does_not_mutate_input_records(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        records = self._records()
        original = [dict(record) for record in records]
        aggregate_percentiles(
            records,
            metrics=("decode_to_output_e2e_ms",),
            warmup_frames=2,
            percentiles=(50,),
        )

        self.assertEqual(records, original)

    def test_warmup_cannot_consume_all_samples(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        with self.assertRaises(ValueError):
            aggregate_percentiles(
                self._records(),
                metrics=("decode_to_output_e2e_ms",),
                warmup_frames=10,
                percentiles=(50, 95),
            )

    def test_negative_warmup_is_rejected(self):
        from on_device.motivation_m1.aggregation import aggregate_percentiles

        with self.assertRaises(ValueError):
            aggregate_percentiles(
                self._records(),
                metrics=("decode_to_output_e2e_ms",),
                warmup_frames=-1,
                percentiles=(50, 95),
            )


if __name__ == "__main__":
    unittest.main()
