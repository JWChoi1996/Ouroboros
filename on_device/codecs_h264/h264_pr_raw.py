from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from custom_h264 import custom_h264 as native


@dataclass(frozen=True)
class CodecTiming:
    input_conversion_ms: float
    encode_ms: float
    decode_ms: float
    rgb_materialization_ms: float
    mv_export_ms: float
    residual_export_ms: float


@dataclass(frozen=True)
class H264Residual:
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    y_valid: np.ndarray
    u_valid: np.ndarray
    v_valid: np.ndarray
    prediction_yuv: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    pre_filter_yuv: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class H264FrameResult:
    packet: bytes
    reconstructed_bgr: np.ndarray
    motion_vectors: np.ndarray | None
    residual: H264Residual | None
    coefficient_summaries: np.ndarray | None
    timing: CodecTiming


class OnlineH264Codec:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        export_mvs: bool = False,
        export_residual: bool = False,
        export_validation: bool = False,
        x264_params: str | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("YUV420P width and height must be positive and even.")
        if export_validation and not export_residual:
            raise ValueError("Validation planes require residual export.")
        self.width = int(width)
        self.height = int(height)
        self.export_mvs = bool(export_mvs)
        self.export_residual = bool(export_residual)
        self.export_validation = bool(export_validation)
        self._closed = False
        status = native.h264_full_init(
            self.width,
            self.height,
            int(fps),
            export_mvs=self.export_mvs,
            export_residual=self.export_residual,
            export_validation=self.export_validation,
            x264_params=x264_params,
        )
        if status != 0:
            self._closed = True
            raise RuntimeError(f"H.264 codec initialization failed with status {status}.")

    def process(self, frame: np.ndarray) -> H264FrameResult:
        if self._closed:
            raise RuntimeError("H.264 codec is closed.")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected BGR frame shape {(self.height, self.width, 3)}, "
                f"got {frame.shape}."
            )
        packet, reconstructed, vectors, raw_residual, raw_timing = (
            native.h264_full_process(
                frame,
                export_mvs=self.export_mvs,
                export_residual=self.export_residual,
                export_validation=self.export_validation,
            )
        )
        residual = None
        summaries = None
        if raw_residual is not None:
            (
                y, u, v, y_valid, u_valid, v_valid,
                predictions, reconstructions, summaries,
            ) = raw_residual
            residual = H264Residual(
                y=y,
                u=u,
                v=v,
                y_valid=y_valid.astype(bool, copy=False),
                u_valid=u_valid.astype(bool, copy=False),
                v_valid=v_valid.astype(bool, copy=False),
                prediction_yuv=predictions,
                pre_filter_yuv=reconstructions,
            )
        scale = 1e-6
        timing = CodecTiming(
            input_conversion_ms=raw_timing.input_conversion_ns * scale,
            encode_ms=raw_timing.encode_ns * scale,
            decode_ms=raw_timing.decode_ns * scale,
            rgb_materialization_ms=raw_timing.rgb_materialization_ns * scale,
            mv_export_ms=raw_timing.mv_export_ns * scale,
            residual_export_ms=raw_timing.residual_export_ns * scale,
        )
        return H264FrameResult(
            packet=packet,
            reconstructed_bgr=reconstructed,
            motion_vectors=vectors,
            residual=residual,
            coefficient_summaries=summaries,
            timing=timing,
        )

    def close(self) -> None:
        if not self._closed:
            native.h264_full_close()
            self._closed = True

    def __enter__(self) -> OnlineH264Codec:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
