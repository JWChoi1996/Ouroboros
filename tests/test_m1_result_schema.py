import hashlib
import unittest


DIRTY_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05)


class M1ResultSchemaTests(unittest.TestCase):
    @staticmethod
    def _record(method="B0", provenance="NATIVE", dirty_ratio=0.5):
        stages = {
            "input_conversion": 99.0,
            "encode": 1.0,
            "decode": 2.0,
            "rgb_materialization": 3.0,
            "mv_export": 0.0,
            "residual_export": 0.0,
            "affine": 0.0,
            "warp": 0.0,
            "diff": 0.0,
            "mask": 0.0,
            "model_preprocess": 4.0,
            "model_forward": 5.0,
            "model_diagnostics": {},
        }
        record = {
            "schema_version": 1,
            "sequence": "seq",
            "frame_index": 21,
            "method": method,
            "provenance": provenance,
            "method_lineage": {
                "B0": "NATIVE_DENSE",
                "B1-MV": "NATIVE_MV_LINEAGE",
                "B2": "PRIOR_WORK_EVENTFUL",
                "B3": "PRIOR_WORK_MASKVD",
                "B4": "PRIOR_WORK_STGT",
            }.get(method, "UNKNOWN"),
            "measurement_fidelity": {
                "B0": "NATIVE",
                "B1-MV": "ADAPTED_DECODER_MV",
                "B2": "ADAPTED_FORCED_MASK",
                "B3": "PROXY_FORCED_MASK",
                "B4": "PROXY_FORCED_MASK",
            }.get(method, "UNKNOWN"),
            "mv_origin": "decoder_side_data" if method == "B1-MV" else "disabled",
            "execution_decode_dependency": True,
            "dirty_ratio": dirty_ratio,
            "effective_dirty_ratio": dirty_ratio,
            "repeat": 0,
            "measured": True,
            "phase": "measured",
            "input_sha256": "a" * 64,
            "reconstructed_sha256": "b" * 64,
            "codec": {
                "export_mvs": method == "B1-MV",
                "export_residual": False,
                "packet_size_bytes": 1,
                "mv_count": 1 if method == "B1-MV" else 0,
                "residual_count": 0,
            },
            "stages_ms": stages,
            "totals_ms": {
                "decode_to_output_e2e": 14.0,
                "source_to_output_total": 15.0,
            },
            "cache_bytes": 0,
            "detection_count": 0,
            "natural_dirty_ratio": dirty_ratio if method == "B1-MV" else None,
            "b0_dirty_ratio_ignored": method == "B0",
        }
        if method == "B1-MV":
            record["stages_ms"].update(
                mv_export=1.0,
                affine=1.0,
                warp=1.0,
                diff=1.0,
                mask=1.0,
            )
            record["totals_ms"] = {
                "decode_to_output_e2e": None,
                "source_to_output_total": 15.0,
            }
        return record

    def test_frame_records_allow_only_m1_execution_methods(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        for method, provenance in (("B0", "NATIVE"), ("B1-MV", "ADAPTED")):
            with self.subTest(method=method):
                validate_frame_record(self._record(method, provenance))

        with self.assertRaises(ValueError):
            validate_frame_record(self._record("UNKNOWN", "NATIVE"))

    def test_b1_of_is_forbidden_as_an_m1_frame_method(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        with self.assertRaises(ValueError):
            validate_frame_record(self._record("B1-OF", "NATIVE"))

    def test_prior_work_frame_records_require_adapted_or_proxy_provenance(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        expected = {"B2": "ADAPTED", "B3": "PROXY", "B4": "PROXY"}
        for method, valid_tag in expected.items():
            validate_frame_record(self._record(method, valid_tag))
            for invalid_tag in {"NATIVE", "ADAPTED", "PROXY"} - {valid_tag}:
                with self.subTest(method=method, tag=invalid_tag), self.assertRaises(
                    ValueError
                ):
                    validate_frame_record(self._record(method, invalid_tag))

    def test_dirty_ratio_must_be_one_of_the_protocol_values(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        for ratio in DIRTY_RATIOS:
            with self.subTest(ratio=ratio):
                validate_frame_record(self._record(dirty_ratio=ratio))

        for ratio in (0.0, 0.2, 0.9, 1.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                validate_frame_record(self._record(dirty_ratio=ratio))

    def test_phase_measured_and_priming_ratio_are_consistent(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        record = self._record()
        record.update(frame_index=0, phase="priming", measured=False)
        record["effective_dirty_ratio"] = 1.0
        validate_frame_record(record)

        record["measured"] = True
        with self.assertRaisesRegex(ValueError, "phase.*measured"):
            validate_frame_record(record)

    def test_stage_totals_follow_accounting_and_exclude_input_conversion(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        record = self._record()
        validate_frame_record(record)
        record["totals_ms"]["source_to_output_total"] += 99.0
        with self.assertRaisesRegex(ValueError, "accounting"):
            validate_frame_record(record)

    def test_b1_requires_decoder_mv_fidelity_metadata(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        record = self._record("B1-MV", "ADAPTED")
        validate_frame_record(record)
        record["mv_origin"] = "encoder"
        with self.assertRaisesRegex(ValueError, "mv_origin"):
            validate_frame_record(record)

    def test_non_b1_natural_dirty_ratio_is_strict_json_null(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        record = self._record()
        validate_frame_record(record)
        record["natural_dirty_ratio"] = record["dirty_ratio"]
        with self.assertRaisesRegex(ValueError, "natural_dirty_ratio"):
            validate_frame_record(record)

    def test_required_field_types_are_enforced(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        record = self._record()
        del record["input_sha256"]
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            validate_frame_record(record)

        record = self._record()
        record["repeat"] = True
        with self.assertRaisesRegex(ValueError, "repeat"):
            validate_frame_record(record)

        for field in ("repeat", "cache_bytes", "detection_count"):
            record = self._record()
            record[field] = -1
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                validate_frame_record(record)

    def test_ep_vit_and_fluxshard_are_not_frame_methods(self):
        from on_device.motivation_m1.result_schema import validate_frame_record

        for method in ("EP-ViT", "FluxShard"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                validate_frame_record(self._record(method, "UNAVAILABLE"))

    def test_manifest_represents_deferred_methods_as_availability_entries(self):
        from on_device.motivation_m1.result_schema import validate_run_manifest

        manifest = self._manifest()

        validate_run_manifest(manifest)

    def test_manifest_requires_reproducibility_and_both_equivalence_gates(self):
        from on_device.motivation_m1.result_schema import validate_run_manifest

        required_paths = (
            ("source_provenance",),
            ("checkpoint", "load_missing_keys"),
            ("codec", "backend", "available"),
            ("instrumentation_equivalence",),
            ("dense_endpoint_equivalence",),
            ("dense_endpoint_status",),
        )
        for path in required_paths:
            manifest = self._manifest()
            target = manifest
            for name in path[:-1]:
                target = target[name]
            del target[path[-1]]
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_run_manifest(manifest)

    def test_manifest_requires_exact_source_set_and_canonical_digest(self):
        from on_device.motivation_m1.result_schema import (
            REQUIRED_M1_SOURCE_PATHS,
            validate_run_manifest,
        )

        manifest = self._manifest()
        files = manifest["source_provenance"]["source_manifest"]["files"]
        self.assertEqual(
            {entry["path"] for entry in files},
            set(REQUIRED_M1_SOURCE_PATHS),
        )
        validate_run_manifest(manifest)

        missing = self._manifest()
        missing_files = missing["source_provenance"]["source_manifest"]["files"]
        missing_files.pop()
        missing["source_provenance"]["source_manifest"]["sha256"] = (
            self._manifest_digest(missing_files)
        )
        with self.assertRaisesRegex(ValueError, "exact required M1 source"):
            validate_run_manifest(missing)

        fabricated = self._manifest()
        fabricated["source_provenance"]["source_manifest"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_run_manifest(fabricated)

    def test_manifest_rejects_deferred_methods_marked_as_executed(self):
        from on_device.motivation_m1.result_schema import validate_run_manifest

        for method in ("EP-ViT", "FluxShard"):
            manifest = self._manifest()
            manifest["method_availability"][method]["status"] = "EXECUTED"
            with self.subTest(method=method), self.assertRaises(ValueError):
                validate_run_manifest(manifest)

    def test_manifest_requires_exact_ordered_dirty_ratio_sweep(self):
        from on_device.motivation_m1.result_schema import validate_run_manifest

        manifest = self._manifest()
        manifest["arguments"]["dirty_ratios"] = [1.0, 0.75]
        with self.assertRaisesRegex(ValueError, "dirty_ratios"):
            validate_run_manifest(manifest)

    @staticmethod
    def _manifest():
        from on_device.motivation_m1.result_schema import (
            REQUIRED_M1_SOURCE_PATHS,
        )

        source_files = [
            {"path": path, "sha256": "f" * 64}
            for path in REQUIRED_M1_SOURCE_PATHS
        ]
        return {
            "schema_version": 1,
            "run_id": "run",
            "arguments": {
                "frames": 30,
                "methods": ["B0"],
                "fps": 30,
                "dirty_ratios": [1, 0.75, 0.5, 0.25, 0.1, 0.05],
                "repeats": 1,
                "warmup_frames": 3,
            },
            "sequences": ["s1", "s2"],
            "checkpoint": {
                "sha256": "a" * 64,
                "matched_parameter_byte_ratio": 1.0,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatched_keys": [],
                "load_missing_keys": [],
                "load_unexpected_keys": [],
            },
            "codec": {
                "x264_params": "test",
                "backend": {
                    "available": True,
                    "ffmpeg_tag": "n8.0.1",
                    "ffmpeg_commit": "b" * 40,
                    "libx264_version": "0.test",
                },
            },
            "protocol": {"native_encoder_mv_latency_established": False},
            "source_provenance": {
                "base_git_commit": "c" * 40,
                "git_dirty": True,
                "git_diff_sha256": "d" * 64,
                "untracked_files": [],
                "source_manifest": {
                    "algorithm": "sha256(path NUL content_sha256 LF)",
                    "sha256": M1ResultSchemaTests._manifest_digest(source_files),
                    "files": source_files,
                },
            },
            "instrumentation_equivalence": {"B0": True},
            "dense_endpoint_equivalence": {"B0": True},
            "dense_endpoint_status": {"B0": "EQUIVALENT"},
            "method_availability": {
                "B0": {
                    "status": "EXECUTED",
                    "provenance": "NATIVE",
                    "method_lineage": "NATIVE_DENSE",
                    "measurement_fidelity": "NATIVE",
                    "dense_endpoint_status": "EQUIVALENT",
                },
                "EP-ViT": {"status": "UNAVAILABLE", "reason": "not available"},
                "FluxShard": {"status": "DEFERRED", "reason": "future work"},
                "B1-OF": {"status": "EXCLUDED", "reason": "excluded"},
            },
        }

    @staticmethod
    def _manifest_digest(files):
        digest = hashlib.sha256()
        for entry in sorted(files, key=lambda item: item["path"]):
            digest.update(entry["path"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry["sha256"].encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
