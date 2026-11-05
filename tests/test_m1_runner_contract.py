import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class M1RunnerContractTests(unittest.TestCase):
    def test_parser_defaults_and_repeatable_sequences(self):
        from on_device.motivation_m1.run_m1_vitdet_b import build_parser

        args = build_parser().parse_args(
            ["--sequence", "0001", "--sequence", "0002"]
        )

        self.assertEqual(args.sequence, ["0001", "0002"])
        self.assertEqual(args.frames, 30)
        self.assertEqual(args.dirty_ratios, [1, 0.75, 0.5, 0.25, 0.1, 0.05])
        self.assertEqual(args.methods, ["B0", "B1-MV", "B2", "B3", "B4"])
        self.assertEqual(
            args.checkpoint.parts[-2:], ("dataset", "model_final_61ccd1.pkl")
        )

    def test_parser_rejects_any_frame_count_other_than_thirty(self):
        from on_device.motivation_m1.run_m1_vitdet_b import build_parser

        for count in ("29", "31"):
            with self.subTest(count=count), self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "--sequence",
                        "0001",
                        "--sequence",
                        "0002",
                        "--frames",
                        count,
                    ]
                )

    def test_parser_requires_exact_ordered_full_dirty_ratio_sweep(self):
        from on_device.motivation_m1.run_m1_vitdet_b import build_parser

        invalid = (
            ["1", ".75"],
            [".75", "1", ".5", ".25", ".1", ".05"],
            ["1", ".75", ".5", ".25", ".1", ".1"],
        )
        for ratios in invalid:
            with self.subTest(ratios=ratios), self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "--sequence",
                        "0001",
                        "--sequence",
                        "0002",
                        "--dirty-ratios",
                        *ratios,
                    ]
                )

    def test_priming_and_warmup_measurement_selection(self):
        from on_device.motivation_m1.run_m1_vitdet_b import frame_phase

        self.assertEqual(frame_phase(0, warmup_frames=2), "priming")
        self.assertEqual(frame_phase(1, warmup_frames=2), "warmup")
        self.assertEqual(frame_phase(2, warmup_frames=2), "warmup")
        self.assertEqual(frame_phase(3, warmup_frames=2), "measured")

    def test_manifest_declares_deferred_methods_and_excludes_optical_flow(self):
        from on_device.motivation_m1.run_m1_vitdet_b import build_manifest

        manifest = build_manifest(
            run_id="run",
            args={
                "frames": 30,
                "fps": 30,
                "methods": ["B0", "B1-MV"],
                "dirty_ratios": [1, 0.75, 0.5, 0.25, 0.1, 0.05],
                "repeats": 1,
                "warmup_frames": 3,
            },
            checkpoint={"sha256": "a" * 64},
            sequences=["0001", "0002"],
            backend={"libx264_version": "test"},
        )

        availability = manifest["method_availability"]
        self.assertEqual(availability["EP-ViT"]["status"], "UNAVAILABLE")
        self.assertEqual(availability["FluxShard"]["status"], "DEFERRED")
        self.assertEqual(
            availability["EP-ViT"]["official_code_search_date"], "2026-11-04"
        )
        self.assertEqual(availability["B1-OF"]["status"], "EXCLUDED")
        b1 = availability["B1-MV"]
        self.assertEqual(b1["provenance"], "ADAPTED")
        self.assertEqual(b1["method_lineage"], "NATIVE_MV_LINEAGE")
        self.assertEqual(b1["measurement_fidelity"], "ADAPTED_DECODER_MV")
        self.assertEqual(b1["mv_origin"], "decoder_side_data")
        self.assertTrue(b1["execution_decode_dependency"])
        self.assertFalse(
            manifest["protocol"]["native_encoder_mv_latency_established"]
        )

    def test_strict_json_output_bundle_has_required_files(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            validate_existing_bundle,
            write_output_bundle,
        )

        manifest = self._manifest()
        records = self._record_set(manifest)
        from on_device.motivation_m1.run_m1_vitdet_b import summarize

        with tempfile.TemporaryDirectory() as directory:
            run_dir = write_output_bundle(
                Path(directory),
                "test-run",
                records,
                summarize(records),
                self._environment_record(),
                manifest,
                make_plots=False,
            )

            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "frames.jsonl",
                    "summary.json",
                    "environment.json",
                    "provenance.json",
                },
            )
            decoded = json.loads(
                (run_dir / "frames.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(decoded["method"], "B0")
            evidence = validate_existing_bundle(run_dir)
            self.assertEqual(evidence["records"], 360)
            self.assertEqual(evidence["summary_groups"], 12)

    def test_reconstructed_hash_consistency_checker_groups_rgb_methods(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            check_reconstructed_hash_consistency,
        )

        records = []
        for method in ("B0", "B2", "B3", "B4"):
            record = self._frame_record()
            record.update(
                method=method,
                reconstructed_sha256="same",
                provenance="ADAPTED" if method == "B2" else (
                    "PROXY" if method in {"B3", "B4"} else "NATIVE"
                ),
            )
            records.append(record)
        check_reconstructed_hash_consistency(records)
        records[-1]["reconstructed_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "reconstructed hash"):
            check_reconstructed_hash_consistency(records)

    def test_output_invariants_reject_metadata_and_accounting_tampering(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            _validate_output_invariants,
        )

        valid = self._frame_record()
        corruptions = (
            ("residual", lambda item: item["codec"].update(export_residual=True)),
            ("motion vectors", lambda item: item["codec"].update(mv_count=1)),
            (
                "accounting",
                lambda item: item["totals_ms"].update(
                    source_to_output_total=999.0
                ),
            ),
        )
        for expected, corrupt in corruptions:
            record = copy.deepcopy(valid)
            corrupt(record)
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, expected
            ):
                _validate_output_invariants([record])

    def test_non_mv_diagnostics_are_zero_and_not_computed(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            _motion_diagnostics,
        )

        timer = object()
        with patch(
            "on_device.motivation_m1.run_m1_vitdet_b._natural_mask"
        ) as natural_mask:
            diagnostics = _motion_diagnostics(
                "B2",
                previous=object(),
                current=object(),
                motion_vectors=None,
                timer=timer,
                forced_ratio=0.25,
            )

        natural_mask.assert_not_called()
        self.assertEqual(diagnostics, (None, None))

    def test_summary_stacks_only_accounted_stages(self):
        from on_device.motivation_m1.run_m1_vitdet_b import summarize

        b1 = self._frame_record(method="B1-MV")
        b1.update(
            provenance="ADAPTED",
            method_lineage="NATIVE_MV_LINEAGE",
            measurement_fidelity="ADAPTED_DECODER_MV",
            mv_origin="decoder_side_data",
            execution_decode_dependency=True,
            natural_dirty_ratio=0.4,
        )
        b1["codec"].update(export_mvs=True, mv_count=2)
        b1["totals_ms"] = {
            "decode_to_output_e2e": None,
            "source_to_output_total": 11.0,
        }
        b1["stages_ms"].update(
            encode=1.0,
            mv_export=1.0,
            affine=1.0,
            warp=1.0,
            diff=1.0,
            mask=1.0,
            model_preprocess=2.0,
            model_forward=3.0,
            decode=100.0,
            rgb_materialization=100.0,
        )

        group = summarize([b1])["groups"][0]

        self.assertEqual(
            set(group["stage_mean_ms"]),
            {
                "encode",
                "mv_export",
                "affine",
                "warp",
                "diff",
                "mask",
                "model_preprocess",
                "model_forward",
            },
        )
        self.assertEqual(sum(group["stage_mean_ms"].values()), 11.0)
        self.assertIsNone(group["decode_to_output_e2e_ms"])
        self.assertEqual(
            group["headline_e2e_ms"],
            group["source_to_output_total_ms"],
        )

        b0 = self._frame_record()
        b0["natural_dirty_ratio"] = None
        b0_group = summarize([b0])["groups"][0]
        self.assertIsNone(b0_group["natural_dirty_ratio"])
        self.assertEqual(
            b0_group["decode_to_output_e2e_ms"],
            {"p50": 14.0, "p95": 14.0},
        )
        self.assertEqual(
            b0_group["headline_e2e_ms"],
            b0_group["decode_to_output_e2e_ms"],
        )

    def test_plot_series_keeps_methods_and_sequences_separate(self):
        from on_device.motivation_m1.run_m1_vitdet_b import _plot_series

        groups = [
            {
                "sequence": sequence,
                "method": method,
                "dirty_ratio": ratio,
                "headline_e2e_ms": {"p50": value},
            }
            for sequence, method, ratio, value in (
                ("s1", "B0", 1.0, 10),
                ("s1", "B0", 0.5, 11),
                ("s1", "B2", 1.0, 8),
                ("s2", "B0", 1.0, 12),
            )
        ]

        series = _plot_series(groups)

        self.assertEqual(set(series), {("s1", "B0"), ("s1", "B2"), ("s2", "B0")})
        self.assertEqual(series[("s1", "B0")], [(0.5, 11.0), (1.0, 10.0)])

    def test_plots_render_labeled_stacks_and_separate_series(self):
        from on_device.motivation_m1.run_m1_vitdet_b import _write_plots

        groups = []
        for method, value in (("B0", 10.0), ("B2", 8.0)):
            groups.append(
                {
                    "sequence": "s1",
                    "method": method,
                    "dirty_ratio": 0.5,
                    "source_to_output_total_ms": {"p50": value},
                    "decode_to_output_e2e_ms": {"p50": value - 1, "p95": value},
                    "headline_e2e_ms": {"p50": value - 1, "p95": value},
                    "stage_mean_ms": {
                        "encode": 1.0,
                        "model_forward": value - 1.0,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_plots(output, {"groups": groups})

            self.assertGreater((output / "latency_stacked.png").stat().st_size, 1000)
            self.assertGreater((output / "saturation.png").stat().st_size, 1000)

    def test_environment_reports_accelerator_and_opencv_versions(self):
        from on_device.motivation_m1.run_m1_vitdet_b import _environment

        fake_torch = SimpleNamespace(
            __version__="2.test",
            version=SimpleNamespace(cuda="12.test"),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: 999)
            ),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda index: "Fake GPU",
            ),
        )
        environment = _environment(
            torch_module=fake_torch,
            cv2_module=SimpleNamespace(__version__="4.test"),
        )

        self.assertEqual(environment["torch"], "2.test")
        self.assertEqual(environment["cuda"], "12.test")
        self.assertEqual(environment["cudnn"], 999)
        self.assertEqual(environment["gpus"], ["Fake GPU"])
        self.assertEqual(environment["opencv"], "4.test")
        self.assertRegex(
            environment["source_provenance"]["git_diff_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_backend_provenance_loads_manifest_separator_neutrally(self):
        from on_device.motivation_m1.run_m1_vitdet_b import backend_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "on_device" / "codec_backend" / "backend_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"libx264_version": "0.test"}), encoding="utf-8"
            )

            provenance = backend_provenance(root)

        self.assertEqual(provenance["libx264_version"], "0.test")
        self.assertEqual(
            Path(provenance["manifest_path"]).parts[-3:],
            ("on_device", "codec_backend", "backend_manifest.json"),
        )

    def test_source_provenance_is_deterministic_and_covers_all_m1_sources(self):
        from on_device.motivation_m1.result_schema import (
            REQUIRED_M1_SOURCE_PATHS,
        )
        from on_device.motivation_m1.run_m1_vitdet_b import (
            REPO_ROOT,
            source_provenance,
        )

        first = source_provenance(REPO_ROOT)
        second = source_provenance(REPO_ROOT)

        self.assertEqual(first, second)
        self.assertRegex(first["base_git_commit"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(first["git_dirty"], bool)
        self.assertRegex(first["git_diff_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["source_manifest"]["sha256"], r"^[0-9a-f]{64}$")
        paths = {
            entry["path"] for entry in first["source_manifest"]["files"]
        }
        self.assertEqual(paths, set(REQUIRED_M1_SOURCE_PATHS))
        for entry in first["untracked_files"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("\\", entry["path"])

    def test_workspace_validation_rejects_fabricated_git_diff_digest(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            REPO_ROOT,
            source_provenance,
            validate_source_provenance_against_workspace,
        )

        manifest = self._manifest()
        manifest["source_provenance"] = source_provenance(REPO_ROOT)
        validate_source_provenance_against_workspace(manifest, REPO_ROOT)

        manifest["source_provenance"]["git_diff_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "git_diff_sha256"):
            validate_source_provenance_against_workspace(manifest, REPO_ROOT)

    def test_dense_endpoint_gate_records_actual_status_without_proxy_abort(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            dense_endpoint_equivalence,
        )

        class FakeAdapter:
            def reset_cache(self):
                pass

            def forward(self, method, frame, mask, timer=None):
                value = 1.0 if method in {"B0", "B2"} else 2.0
                return (
                    __import__("numpy").array([[value]], dtype=float),
                    __import__("numpy").array([1]),
                    __import__("numpy").array([value], dtype=float),
                )

        result = dense_endpoint_equivalence(
            FakeAdapter(),
            setup_frame=object(),
            methods=["B0", "B1-MV", "B2", "B3", "B4"],
        )

        self.assertEqual(
            result["status"],
            {
                "B0": "EQUIVALENT",
                "B1-MV": "NOT_COMPARABLE_INPUT",
                "B2": "EQUIVALENT",
                "B3": "NOT_EQUIVALENT",
                "B4": "NOT_EQUIVALENT",
            },
        )
        self.assertEqual(
            result["equivalent"],
            {"B0": True, "B1-MV": None, "B2": True, "B3": False, "B4": False},
        )

    def test_record_set_requires_complete_protocol_cartesian_product(self):
        from on_device.motivation_m1.run_m1_vitdet_b import validate_record_set

        manifest = self._manifest()
        records = self._record_set(manifest)
        validate_record_set(records, manifest)

        with self.assertRaisesRegex(ValueError, "record set"):
            validate_record_set(records[:-1], manifest)

    def test_record_set_requires_effective_ratio_on_every_non_priming_frame(self):
        from on_device.motivation_m1.run_m1_vitdet_b import validate_record_set

        manifest = self._manifest()
        records = self._record_set(manifest)
        target = next(
            record
            for record in records
            if record["dirty_ratio"] == 0.75 and record["frame_index"] == 1
        )
        target["effective_dirty_ratio"] = 1.0

        with self.assertRaisesRegex(ValueError, "effective_dirty_ratio"):
            validate_record_set(records, manifest)

    def test_summary_and_environment_are_validated_before_output(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            summarize,
            validate_environment,
            validate_summary,
        )

        records = self._record_set(self._manifest())
        summary = summarize(records)
        environment = self._environment_record()
        validate_summary(summary)
        validate_environment(environment)

        with self.assertRaisesRegex(ValueError, "groups"):
            validate_summary({"schema_version": 1})
        with self.assertRaisesRegex(ValueError, "python"):
            invalid_environment = dict(environment)
            del invalid_environment["python"]
            validate_environment(invalid_environment)

    def test_summary_is_cross_validated_against_fresh_aggregation(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            summarize,
            validate_summary_against_records,
        )

        records = self._record_set(self._manifest())
        expected = summarize(records)
        validate_summary_against_records(expected, records)

        corruptions = (
            lambda summary: summary["groups"].pop(),
            lambda summary: summary["groups"][0].update(sample_count=999),
            lambda summary: summary["groups"][0][
                "decode_to_output_e2e_ms"
            ].update(p50=999.0),
            lambda summary: summary["groups"][0][
                "source_to_output_total_ms"
            ].update(p95=999.0),
            lambda summary: summary["groups"][0][
                "headline_e2e_ms"
            ].update(p50=999.0),
            lambda summary: summary["groups"][0]["stage_mean_ms"].update(
                encode=999.0
            ),
        )
        for corrupt in corruptions:
            summary = copy.deepcopy(expected)
            corrupt(summary)
            with self.assertRaisesRegex(ValueError, "summary"):
                validate_summary_against_records(summary, records)

    def test_output_rejects_stale_summary_before_writing(self):
        from on_device.motivation_m1.run_m1_vitdet_b import (
            summarize,
            write_output_bundle,
        )

        manifest = self._manifest()
        records = self._record_set(manifest)
        summary = summarize(records)
        summary["groups"][0]["source_to_output_total_ms"]["p50"] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "summary"):
                write_output_bundle(
                    output,
                    "stale",
                    records,
                    summary,
                    self._environment_record(),
                    manifest,
                    make_plots=False,
                )
            self.assertFalse((output / "stale").exists())

    @staticmethod
    def _frame_record(method="B0"):
        stages = {
            "input_conversion": 9.0,
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
        return {
            "schema_version": 1,
            "sequence": "0001",
            "frame_index": 1,
            "method": method,
            "dirty_ratio": 1.0,
            "effective_dirty_ratio": 1.0,
            "repeat": 0,
            "measured": True,
            "phase": "measured",
            "provenance": "NATIVE",
            "method_lineage": "NATIVE_DENSE",
            "measurement_fidelity": "NATIVE",
            "mv_origin": "disabled",
            "execution_decode_dependency": True,
            "input_sha256": "a" * 64,
            "reconstructed_sha256": "b" * 64,
            "codec": {
                "export_mvs": False,
                "export_residual": False,
                "packet_size_bytes": 1,
                "mv_count": 0,
                "residual_count": 0,
            },
            "stages_ms": stages,
            "totals_ms": {
                "decode_to_output_e2e": 14.0,
                "source_to_output_total": 15.0,
            },
            "cache_bytes": 0,
            "natural_dirty_ratio": None,
            "detection_count": 0,
            "b0_dirty_ratio_ignored": True,
        }

    @staticmethod
    def _manifest():
        from on_device.motivation_m1.run_m1_vitdet_b import (
            REPO_ROOT,
            source_provenance,
        )

        if not hasattr(M1RunnerContractTests, "_cached_source_provenance"):
            M1RunnerContractTests._cached_source_provenance = source_provenance(
                REPO_ROOT
            )
        return {
            "schema_version": 1,
            "run_id": "test-run",
            "arguments": {
                "frames": 30,
                "methods": ["B0"],
                "fps": 30,
                "dirty_ratios": [1, 0.75, 0.5, 0.25, 0.1, 0.05],
                "repeats": 1,
                "warmup_frames": 3,
            },
            "sequences": ["0001", "0002"],
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
            "protocol": {
                "first_frame": "unmeasured 100%-dirty codec/cache priming",
                "native_encoder_mv_latency_established": False,
            },
            "source_provenance": copy.deepcopy(
                M1RunnerContractTests._cached_source_provenance
            ),
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
                "FluxShard": {"status": "DEFERRED", "reason": "deferred"},
                "B1-OF": {"status": "EXCLUDED", "reason": "excluded"},
            },
        }

    @classmethod
    def _record_set(cls, manifest):
        records = []
        warmup = manifest["arguments"]["warmup_frames"]
        for sequence in manifest["sequences"]:
            for method in manifest["arguments"]["methods"]:
                for ratio in manifest["arguments"]["dirty_ratios"]:
                    for repeat in range(manifest["arguments"]["repeats"]):
                        for frame_index in range(30):
                            record = copy.deepcopy(cls._frame_record(method))
                            phase = (
                                "priming"
                                if frame_index == 0
                                else "warmup"
                                if frame_index <= warmup
                                else "measured"
                            )
                            record.update(
                                sequence=sequence,
                                frame_index=frame_index,
                                dirty_ratio=float(ratio),
                                effective_dirty_ratio=(
                                    1.0 if frame_index == 0 else float(ratio)
                                ),
                                repeat=repeat,
                                phase=phase,
                                measured=phase == "measured",
                                natural_dirty_ratio=None,
                            )
                            records.append(record)
        return records

    @staticmethod
    def _environment_record():
        return {
            "python": "3.test",
            "platform": "test",
            "numpy": "test",
            "opencv": "test",
            "torch": "test",
            "cuda": None,
            "cudnn": None,
            "cuda_available": False,
            "gpus": [],
            "git_commit": None,
            "source_provenance": copy.deepcopy(
                M1RunnerContractTests._manifest()["source_provenance"]
            ),
        }


if __name__ == "__main__":
    unittest.main()
