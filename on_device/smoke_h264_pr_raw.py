from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ON_DEVICE_ROOT = REPO_ROOT / "on_device"
SMOKE_FRAME_COUNT = 30
DEFAULT_X264_PARAMS = (
    "keyint=9999:min-keyint=9999:no-scenecut=1:bframes=0:"
    "ref=1:partitions=i16x16,p16x16:8x8dct=0:rc-lookahead=0"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def exact_smoke_frame_count(value: str) -> int:
    parsed = int(value)
    if parsed != SMOKE_FRAME_COUNT:
        raise argparse.ArgumentTypeError(
            f"Task 6 smoke requires exactly {SMOKE_FRAME_COUNT} frames"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the stateful H.264 P_raw baseline and diagnostic pass."
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", required=True, type=exact_smoke_frame_count)
    parser.add_argument("--fps", required=True, type=positive_int)
    parser.add_argument("--model", required=True, choices=("vitdet-b",))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def resolve_sequence(
    repository_root: Path, sequence: str, frame_count: int
) -> list[Path]:
    data_root = repository_root / "data" / "vid"
    candidates = [
        data_root / split / "frames" / sequence
        for split in (
            "vid_cocoval",
            "vid_val",
            "vid_minival",
            "vid_train",
            "det_train",
        )
    ]
    if data_root.is_dir():
        candidates.extend(
            path
            for path in data_root.rglob(sequence)
            if path.is_dir() and path.parent.name == "frames"
        )

    seen: set[Path] = set()
    short_matches: list[tuple[Path, int]] = []
    for candidate in candidates:
        try:
            identity = candidate.resolve()
        except OSError:
            identity = candidate
        if identity in seen or not candidate.is_dir():
            continue
        seen.add(identity)
        paths = sorted(
            path
            for path in candidate.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if len(paths) >= frame_count:
            return paths[:frame_count]
        short_matches.append((candidate, len(paths)))

    if short_matches:
        path, available = max(short_matches, key=lambda item: item[1])
        raise ValueError(
            f"Sequence {sequence!r} at {path} requested {frame_count} frames "
            f"but found {available} JPEG frames."
        )
    raise FileNotFoundError(
        f"ImageNet-VID sequence {sequence!r} was not found under {data_root}. "
        "Expected an unpacked split containing frames/<sequence>/*.jpg."
    )


def load_source_frames(paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int, int] | None = None
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unreadable JPEG frame: {path}")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected uint8 BGR JPEG frame, got {frame.shape}: {path}")
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError(
                f"H.264 YUV420 requires even dimensions, got {width}x{height}: {path}"
            )
        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Changing frame dimensions are unsupported: expected "
                f"{expected_shape[1]}x{expected_shape[0]}, got {width}x{height}: {path}"
            )
        frames.append(np.ascontiguousarray(frame))
    return frames


def annex_b_nal_units(packet: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index < len(packet) - 2:
        if packet[index : index + 4] == b"\x00\x00\x00\x01":
            header_index = index + 4
        elif packet[index : index + 3] == b"\x00\x00\x01":
            header_index = index + 3
        else:
            index += 1
            continue
        if header_index >= len(packet):
            raise ValueError("Annex-B start code is missing its NAL header.")
        starts.append((index, header_index))
        index = header_index + 1
    return [
        packet[header_index : starts[position + 1][0] if position + 1 < len(starts) else len(packet)]
        for position, (_, header_index) in enumerate(starts)
    ]


def annex_b_nal_unit_types(packet: bytes) -> list[int]:
    return [nal_unit[0] & 0x1F for nal_unit in annex_b_nal_units(packet)]


def rbsp_from_ebsp(ebsp: bytes) -> bytes:
    rbsp = bytearray()
    index = 0
    while index < len(ebsp):
        if ebsp[index : index + 3] == b"\x00\x00\x03":
            if index + 3 >= len(ebsp) or ebsp[index + 3] > 0x03:
                raise ValueError("Malformed H.264 emulation-prevention sequence.")
            rbsp.extend(b"\x00\x00")
            index += 3
            continue
        rbsp.append(ebsp[index])
        index += 1
    return bytes(rbsp)


def _read_unsigned_exp_golomb(rbsp: bytes, bit_offset: int) -> tuple[int, int]:
    bit_count = len(rbsp) * 8

    def read_bit(offset: int) -> int:
        if offset >= bit_count:
            raise ValueError("Truncated or malformed H.264 slice-header RBSP.")
        return (rbsp[offset // 8] >> (7 - offset % 8)) & 1

    leading_zero_bits = 0
    while read_bit(bit_offset) == 0:
        leading_zero_bits += 1
        bit_offset += 1
    bit_offset += 1
    suffix = 0
    for _ in range(leading_zero_bits):
        suffix = (suffix << 1) | read_bit(bit_offset)
        bit_offset += 1
    return (1 << leading_zero_bits) - 1 + suffix, bit_offset


def _slice_type_from_nal_unit(nal_unit: bytes) -> int:
    if len(nal_unit) < 2:
        raise ValueError("Truncated or malformed H.264 slice-header RBSP.")
    rbsp = rbsp_from_ebsp(nal_unit[1:])
    _, bit_offset = _read_unsigned_exp_golomb(rbsp, 0)
    slice_type, _ = _read_unsigned_exp_golomb(rbsp, bit_offset)
    return slice_type % 5


def packet_frame_metadata(packet: bytes) -> dict[str, Any]:
    nal_units = annex_b_nal_units(packet)
    if not nal_units:
        raise ValueError("Packet is not an Annex-B H.264 access unit.")
    nal_types = [nal_unit[0] & 0x1F for nal_unit in nal_units]
    slice_types: list[int] = []
    slice_type_names: list[str] = []
    for nal_unit, nal_type in zip(nal_units, nal_types, strict=True):
        if nal_type not in {1, 5}:
            continue
        slice_type = _slice_type_from_nal_unit(nal_unit)
        slice_name = ("P", "B", "I", "SP", "SI")[slice_type]
        if nal_type == 5 and slice_type != 2:
            raise ValueError(
                f"IDR NAL type 5 must carry an I slice, got {slice_name}."
            )
        if slice_type in {1, 3, 4}:
            raise ValueError(f"unsupported {slice_name} slice in IP-only codec packet.")
        slice_types.append(slice_type)
        slice_type_names.append(slice_name)
    if not slice_types:
        raise ValueError(
            f"Annex-B packet is missing coded slice NAL type 1 or 5: {nal_types}"
        )
    if len(set(slice_types)) != 1:
        raise ValueError(
            f"Annex-B packet has ambiguous coded slice types: {slice_type_names}"
        )
    return {
        "nal_unit_types": nal_types,
        "slice_types": slice_types,
        "slice_type_names": slice_type_names,
        "frame_type": slice_type_names[0],
    }


def latency_totals(timing: Any, model_total_ms: float) -> dict[str, float]:
    decode_to_output = (
        float(timing.decode_ms)
        + float(timing.rgb_materialization_ms)
        + float(model_total_ms)
    )
    return {
        "decode_to_output_e2e_ms": decode_to_output,
        "source_to_output_total_ms": float(timing.encode_ms) + decode_to_output,
    }


def residual_statistics(residual: Any) -> dict[str, int | float]:
    valid_values = np.concatenate(
        [
            plane[valid].astype(np.int32, copy=False)
            for plane, valid in (
                (residual.y, residual.y_valid),
                (residual.u, residual.u_valid),
                (residual.v, residual.v_valid),
            )
        ]
    )
    valid_count = int(valid_values.size)
    nonzero_count = int(np.count_nonzero(valid_values))
    return {
        "residual_valid_count": valid_count,
        "residual_nonzero_count": nonzero_count,
        "residual_nonzero_ratio": (
            float(nonzero_count / valid_count) if valid_count else 0.0
        ),
        "residual_min": int(valid_values.min()) if valid_count else 0,
        "residual_max": int(valid_values.max()) if valid_count else 0,
    }


def coefficient_summary_metadata(summaries: np.ndarray) -> dict[str, int]:
    return {"coefficient_summary_row_count": int(summaries.shape[0])}


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {"frame_count": len(records)}
    if not records:
        return aggregates
    common_keys = set.intersection(*(set(record) for record in records))
    for key in sorted(common_keys):
        values = [record[key] for record in records]
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
            for value in values
        ):
            numeric = [float(value) for value in values]
            total = sum(numeric)
            if all(isinstance(value, (int, np.integer)) for value in values):
                total = int(total)
            aggregates[key] = {
                "sum": total,
                "mean": float(sum(numeric) / len(numeric)),
                "min": min(numeric),
                "max": max(numeric),
            }
    return aggregates


def reconstructed_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def source_reconstruction_psnr(source: np.ndarray, reconstructed: np.ndarray) -> float:
    difference = source.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def prepare_vitdet_input(reconstructed_bgr: np.ndarray) -> Any:
    import torch
    import torchvision.transforms.functional as functional
    from torchvision import transforms

    reconstructed_rgb = np.ascontiguousarray(reconstructed_bgr[:, :, ::-1])
    image = torch.from_numpy(reconstructed_rgb).permute(2, 0, 1)
    height, width = image.shape[-2:]
    scale = min(640 / min(height, width), int(1024 * 0.9) / max(height, width))
    if scale != 1.0:
        image = functional.resize(
            image,
            [round(scale * height), round(scale * width)],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    return image


def infer_reconstructed_bgr(
    model: Any,
    reconstructed_bgr: np.ndarray,
    *,
    synchronize: Callable[[], None],
    preprocess: Callable[[np.ndarray], Any] = prepare_vitdet_input,
) -> dict[str, Any]:
    import torch

    preprocess_started = time.perf_counter_ns()
    image = preprocess(reconstructed_bgr)
    preprocess_ms = (time.perf_counter_ns() - preprocess_started) * 1e-6
    height, width = image.shape[-2:]
    inputs = [{"image": image, "height": height, "width": width}]
    synchronize()
    forward_started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.base_model(inputs)
    synchronize()
    forward_ms = (time.perf_counter_ns() - forward_started) * 1e-6
    model_total_ms = preprocess_ms + forward_ms
    return {
        "model_preprocess_ms": preprocess_ms,
        "model_forward_ms": forward_ms,
        "model_total_ms": model_total_ms,
        "detection_count": int(len(output[0]["instances"])),
    }


def timing_record(timing: Any) -> dict[str, float]:
    return {
        "input_conversion_ms": float(timing.input_conversion_ms),
        "encode_ms": float(timing.encode_ms),
        "decode_ms": float(timing.decode_ms),
        "rgb_materialization_ms": float(timing.rgb_materialization_ms),
        "mv_export_ms": float(timing.mv_export_ms),
        "residual_export_ms": float(timing.residual_export_ms),
    }


def git_metadata(repository_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def backend_metadata(repository_root: Path) -> dict[str, Any]:
    manifest_path = (
        repository_root / "on_device" / "codec_backend" / "backend_manifest.json"
    )
    metadata: dict[str, Any] = {"manifest_path": str(manifest_path.relative_to(repository_root))}
    if manifest_path.is_file():
        metadata.update(json.loads(manifest_path.read_text(encoding="utf-8")))
    patch_path = (
        repository_root
        / "on_device"
        / "codec_backend"
        / "ffmpeg-8.0.1-ouroboros-residual.patch"
    )
    if patch_path.is_file() and "patch_sha256" not in metadata:
        metadata["patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    return metadata


def load_model(model_name: str) -> Any:
    if str(ON_DEVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(ON_DEVICE_ROOT))
    import torch
    from evaluate_funcs import prepare_environment

    if not torch.cuda.is_available():
        raise RuntimeError("The ViTDet-B smoke measurement requires CUDA.")
    setup = SimpleNamespace(
        model=model_name,
        device="cuda:0",
        dataset="imnet-vid",
        sequence=None,
    )
    model, _dataset, _settings = prepare_environment(setup)
    model.eval()
    return model


def run_baseline(
    frames: list[np.ndarray], paths: list[Path], fps: int, model: Any
) -> list[dict[str, Any]]:
    import torch
    from codecs_h264.h264_pr_raw import OnlineH264Codec

    height, width = frames[0].shape[:2]
    records: list[dict[str, Any]] = []
    with OnlineH264Codec(
        width, height, fps, export_mvs=False, export_residual=False
    ) as codec:
        for index, (source, path) in enumerate(zip(frames, paths, strict=True)):
            result = codec.process(source)
            if not result.packet:
                raise RuntimeError(f"Baseline produced an empty packet at frame {index}.")
            model_record = infer_reconstructed_bgr(
                model, result.reconstructed_bgr, synchronize=torch.cuda.synchronize
            )
            record = {
                "frame_index": index,
                **packet_frame_metadata(result.packet),
                "source_frame": str(path.relative_to(REPO_ROOT)),
                "packet_size_bytes": len(result.packet),
                "reconstructed_sha256": reconstructed_sha256(
                    result.reconstructed_bgr
                ),
                "source_reconstruction_psnr_db": source_reconstruction_psnr(
                    source, result.reconstructed_bgr
                ),
                **timing_record(result.timing),
                **model_record,
            }
            record.update(
                latency_totals(result.timing, model_record["model_total_ms"])
            )
            records.append(record)
    return records


def run_diagnostic(
    frames: list[np.ndarray], paths: list[Path], fps: int
) -> list[dict[str, Any]]:
    from codecs_h264.h264_pr_raw import OnlineH264Codec

    height, width = frames[0].shape[:2]
    records: list[dict[str, Any]] = []
    with OnlineH264Codec(
        width, height, fps, export_mvs=True, export_residual=True
    ) as codec:
        for index, (source, path) in enumerate(zip(frames, paths, strict=True)):
            result = codec.process(source)
            if not result.packet:
                raise RuntimeError(f"Diagnostic produced an empty packet at frame {index}.")
            if result.residual is None or result.coefficient_summaries is None:
                raise RuntimeError(f"Missing exact residual payload at frame {index}.")
            vectors = result.motion_vectors
            mv_count = 0 if vectors is None else int(len(vectors))
            past_reference_mv_count = (
                0 if vectors is None else int(np.count_nonzero(vectors[:, 7] < 0))
            )
            record = {
                "frame_index": index,
                **packet_frame_metadata(result.packet),
                "source_frame": str(path.relative_to(REPO_ROOT)),
                "packet_size_bytes": len(result.packet),
                "reconstructed_sha256": reconstructed_sha256(
                    result.reconstructed_bgr
                ),
                "motion_vector_count": mv_count,
                "past_reference_motion_vector_count": past_reference_mv_count,
                **coefficient_summary_metadata(result.coefficient_summaries),
                **residual_statistics(result.residual),
                **timing_record(result.timing),
            }
            records.append(record)
    return records


def validate_passes(
    baseline: list[dict[str, Any]], diagnostic: list[dict[str, Any]], expected: int
) -> dict[str, Any]:
    if expected != SMOKE_FRAME_COUNT:
        raise RuntimeError(
            f"Task 6 validation requires exactly {SMOKE_FRAME_COUNT} frames."
        )
    if len(baseline) != expected or len(diagnostic) != expected:
        raise RuntimeError(
            f"Expected {expected} records per pass, got "
            f"{len(baseline)} baseline and {len(diagnostic)} diagnostic."
        )
    if any(record["packet_size_bytes"] <= 0 for record in baseline + diagnostic):
        raise RuntimeError("Every encoded packet must be non-empty.")
    expected_frame_types = ["I"] + ["P"] * (SMOKE_FRAME_COUNT - 1)
    for pass_name, records in (("baseline", baseline), ("diagnostic", diagnostic)):
        actual_frame_types = [record["frame_type"] for record in records]
        if actual_frame_types != expected_frame_types:
            raise RuntimeError(
                f"{pass_name} must contain I followed by 29 P frames; "
                f"got {actual_frame_types}."
            )
    if diagnostic[0]["past_reference_motion_vector_count"] > 0:
        raise RuntimeError("The I-frame exported an unexpected past-reference MV.")
    if any(record["motion_vector_count"] <= 0 for record in diagnostic[1:]):
        raise RuntimeError("Every diagnostic P-frame must export motion vectors.")
    if not any(record["residual_nonzero_count"] > 0 for record in diagnostic):
        raise RuntimeError("Diagnostic pass exported no non-zero exact residual samples.")
    hashes_match = all(
        plain["reconstructed_sha256"] == instrumented["reconstructed_sha256"]
        for plain, instrumented in zip(baseline, diagnostic, strict=True)
    )
    if not hashes_match:
        raise RuntimeError(
            "Metadata export changed reconstructed pixels between codec passes."
        )
    return {
        "expected_frame_count": expected,
        "baseline_frame_count": len(baseline),
        "diagnostic_frame_count": len(diagnostic),
        "all_packets_nonempty": True,
        "baseline_frame_types": expected_frame_types,
        "diagnostic_frame_types": expected_frame_types,
        "i_frame_has_no_past_reference_mvs": True,
        "all_p_frames_have_motion_vectors": True,
        "has_nonzero_exact_residual": True,
        "reconstructed_hashes_identical_between_passes": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame_paths = resolve_sequence(REPO_ROOT, args.sequence, args.frames)
    frames = load_source_frames(frame_paths)
    git_info = git_metadata(REPO_ROOT)
    model = load_model(args.model)
    baseline = run_baseline(frames, frame_paths, args.fps, model)
    del model
    diagnostic = run_diagnostic(frames, frame_paths, args.fps)
    validation = validate_passes(baseline, diagnostic, args.frames)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_info,
        "backend": backend_metadata(REPO_ROOT),
        "codec": {
            "name": "H.264/libx264",
            "fps": args.fps,
            "width": int(frames[0].shape[1]),
            "height": int(frames[0].shape[0]),
            "pixel_format": "yuv420p",
            "preset": "fast",
            "profile": "baseline",
            "tune": "zerolatency",
            "x264_params": DEFAULT_X264_PARAMS,
            "baseline_export_mvs": False,
            "baseline_export_residual": False,
            "diagnostic_export_mvs": True,
            "diagnostic_export_residual": True,
        },
        "input": {
            "kind": "imagenet_vid_jpeg_sequence",
            "sequence": args.sequence,
            "requested_frames": args.frames,
            "frame_paths": [
                str(path.relative_to(REPO_ROOT)) for path in frame_paths
            ],
        },
        "model": {
            "name": args.model,
            "wrapper": "MaskedRCNN_ViT_B_FPN_Contexted",
            "input_contract": "codec-reconstructed BGR converted to RGB CHW uint8",
            "cuda_synchronized_timing": True,
            "model_forward_contract": (
                "official base_model forward including detector postprocessing"
            ),
        },
        "baseline": {
            "frames": baseline,
            "aggregates": aggregate_records(baseline),
        },
        "diagnostic": {
            "frames": diagnostic,
            "aggregates": aggregate_records(diagnostic),
        },
        "validation": validation,
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "frames": manifest["validation"]["baseline_frame_count"],
        "mean_decode_to_output_e2e_ms": manifest["baseline"]["aggregates"][
            "decode_to_output_e2e_ms"
        ]["mean"],
        "hashes_match": manifest["validation"][
            "reconstructed_hashes_identical_between_passes"
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
