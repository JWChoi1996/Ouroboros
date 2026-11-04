# custom_h264.py ― custom_h264.c의 Python 래퍼

import ctypes
import numpy as np
import os

# MVInfo 구조체 정의 (custom_h264.c와 동일)
class MVInfo(ctypes.Structure):
    _fields_ = [
        ("dst_x", ctypes.c_int16),
        ("dst_y", ctypes.c_int16),
        ("motion_x", ctypes.c_int32),
        ("motion_y", ctypes.c_int32),
        ("motion_scale", ctypes.c_int32),
    ]

class MVInfoV2(ctypes.Structure):
    _fields_ = [
        ("dst_x", ctypes.c_int16),
        ("dst_y", ctypes.c_int16),
        ("w", ctypes.c_uint8),
        ("h", ctypes.c_uint8),
        ("motion_x", ctypes.c_int32),
        ("motion_y", ctypes.c_int32),
        ("motion_scale", ctypes.c_int32),
        ("source", ctypes.c_int32),
        ("flags", ctypes.c_uint64),
    ]

class CodecTiming(ctypes.Structure):
    _fields_ = [
        ("input_conversion_ns", ctypes.c_uint64),
        ("encode_ns", ctypes.c_uint64),
        ("decode_ns", ctypes.c_uint64),
        ("rgb_materialization_ns", ctypes.c_uint64),
        ("mv_export_ns", ctypes.c_uint64),
        ("residual_export_ns", ctypes.c_uint64),
    ]

class H264ResidualOutput(ctypes.Structure):
    _fields_ = [
        ("y", ctypes.POINTER(ctypes.c_int16)),
        ("u", ctypes.POINTER(ctypes.c_int16)),
        ("v", ctypes.POINTER(ctypes.c_int16)),
        ("y_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("u_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("v_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("prediction_y", ctypes.POINTER(ctypes.c_uint8)),
        ("prediction_u", ctypes.POINTER(ctypes.c_uint8)),
        ("prediction_v", ctypes.POINTER(ctypes.c_uint8)),
        ("reconstruction_y", ctypes.POINTER(ctypes.c_uint8)),
        ("reconstruction_u", ctypes.POINTER(ctypes.c_uint8)),
        ("reconstruction_v", ctypes.POINTER(ctypes.c_uint8)),
        ("coefficient_summaries", ctypes.POINTER(ctypes.c_float)),
        ("y_capacity", ctypes.c_int),
        ("chroma_capacity", ctypes.c_int),
        ("summary_row_capacity", ctypes.c_int),
        ("summary_row_count", ctypes.c_int),
        ("has_validation", ctypes.c_int),
    ]


H264_EXPORT_MVS = 1 << 0
H264_EXPORT_RESIDUAL = 1 << 1
H264_EXPORT_VALIDATION = 1 << 2

# libcustom_h264.so 동적 라이브러리 로드
_lib_path = os.path.join(os.path.dirname(__file__), "libcustom_h264.so")
lib = ctypes.CDLL(_lib_path)

# 함수 시그니처 지정
lib.mv_init.argtypes    = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
lib.mv_init.restype     = ctypes.c_int
lib.mv_process.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(MVInfo), ctypes.c_int]
lib.mv_process.restype  = ctypes.c_int
lib.mv_close.argtypes   = []
lib.mv_close.restype    = None
lib.mv_process_and_encode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(MVInfo), ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),   # encoded_size_out
    ctypes.POINTER(ctypes.c_int)    # mv_cnt_out
]
lib.mv_process_and_encode.restype = ctypes.c_int
lib.codec_init.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
]
lib.codec_init.restype = ctypes.c_int
lib.codec_process.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
    ctypes.POINTER(MVInfoV2), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(CodecTiming),
]
lib.codec_process.restype = ctypes.c_int
lib.h264_full_init.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
]
lib.h264_full_init.restype = ctypes.c_int
lib.h264_full_process.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
    ctypes.POINTER(MVInfoV2), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(H264ResidualOutput),
    ctypes.POINTER(CodecTiming),
]
lib.h264_full_process.restype = ctypes.c_int
lib.h264_full_close.argtypes = []
lib.h264_full_close.restype = None

def mv_init(width: int, height: int, fps: int, x264_params: str = None) -> int:
    """라이브러리 초기화. x264_params를 지정하면 ffmpeg/x264 옵션을 직접 전달."""
    if x264_params is None:
        x264_params = ""
    return lib.mv_init(width, height, fps, x264_params.encode("utf-8"))

def mv_process(frame: np.ndarray, max_mv: int = 8192):
    """프레임(BGR np.ndarray)에서 모션 벡터 추출"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame은 uint8 BGR 3채널이어야 합니다.")
    mv_buf = (MVInfo * max_mv)()
    ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    n = lib.mv_process(ptr, mv_buf, max_mv)
    if n <= 0:
        return np.zeros((0, 5), dtype=np.int32)
    arr = np.zeros((n, 5), dtype=np.int32)
    for i in range(n):
        arr[i, 0] = mv_buf[i].dst_x
        arr[i, 1] = mv_buf[i].dst_y
        arr[i, 2] = mv_buf[i].motion_x
        arr[i, 3] = mv_buf[i].motion_y
        arr[i, 4] = mv_buf[i].motion_scale
    return arr

def mv_process_and_encode(frame: np.ndarray, max_mv: int = 8192, max_bytes: int = 2*1024*1024):
    """프레임(BGR)에서 모션벡터와 H.264 인코딩 바이트를 동시에 추출"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame은 uint8 BGR 3채널이어야 합니다.")
    mv_buf = (MVInfo * max_mv)()
    out_buf = (ctypes.c_uint8 * max_bytes)()
    ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    enc_size = ctypes.c_int()
    mv_cnt   = ctypes.c_int()
    ret = lib.mv_process_and_encode(ptr, mv_buf, max_mv,
                                    out_buf, max_bytes,
                                    ctypes.byref(enc_size), ctypes.byref(mv_cnt))
    if ret != 0:
        raise RuntimeError("encode error")
    encoded_bytes = bytes(out_buf[:enc_size.value])
    arr = np.zeros((mv_cnt.value, 5), dtype=np.int32) if mv_cnt.value > 0 else np.zeros((0, 5), dtype=np.int32)
    for i in range(mv_cnt.value):
        arr[i, 0] = mv_buf[i].dst_x
        arr[i, 1] = mv_buf[i].dst_y
        arr[i, 2] = mv_buf[i].motion_x
        arr[i, 3] = mv_buf[i].motion_y
        arr[i, 4] = mv_buf[i].motion_scale
    return encoded_bytes, arr

def codec_init(width: int, height: int, fps: int, export_mvs: bool = False,
               x264_params: str = None) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("YUV420P width and height must be positive and even.")
    params = "" if x264_params is None else x264_params
    return lib.codec_init(
        width, height, fps, params.encode("utf-8"), int(export_mvs)
    )

def codec_process(frame: np.ndarray, export_mvs: bool = False,
                  max_mv: int = 8192, max_bytes: int = 2 * 1024 * 1024):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a uint8 HWC BGR array.")
    if not frame.flags.c_contiguous:
        raise ValueError("frame must be C-contiguous.")

    height, width = frame.shape[:2]
    output = (ctypes.c_uint8 * max_bytes)()
    reconstructed = np.empty_like(frame)
    mv_buffer = (MVInfoV2 * max_mv)()
    encoded_size = ctypes.c_int()
    mv_count = ctypes.c_int()
    timing = CodecTiming()
    status = lib.codec_process(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output, max_bytes, ctypes.byref(encoded_size),
        reconstructed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width * height * 3,
        mv_buffer, max_mv, ctypes.byref(mv_count),
        ctypes.byref(timing),
    )
    if status != 0:
        raise RuntimeError(f"H.264 process failed with status {status}.")

    vectors = None
    if export_mvs:
        vectors = np.zeros((mv_count.value, 9), dtype=np.float32)
        for index in range(mv_count.value):
            item = mv_buffer[index]
            vectors[index] = (
                item.dst_x - item.w / 2.0,
                item.dst_y - item.h / 2.0,
                item.w,
                item.h,
                item.motion_x,
                item.motion_y,
                item.motion_scale,
                item.source,
                item.flags,
            )
    return bytes(output[:encoded_size.value]), reconstructed, vectors, timing


def h264_full_init(
    width: int,
    height: int,
    fps: int,
    *,
    export_mvs: bool = False,
    export_residual: bool = False,
    export_validation: bool = False,
    x264_params: str = None,
) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("YUV420P width and height must be positive and even.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if export_validation and not export_residual:
        raise ValueError("Validation planes require residual export.")
    flags = (
        (H264_EXPORT_MVS if export_mvs else 0)
        | (H264_EXPORT_RESIDUAL if export_residual else 0)
        | (H264_EXPORT_VALIDATION if export_validation else 0)
    )
    params = "" if x264_params is None else x264_params
    return lib.h264_full_init(
        width, height, fps, params.encode("utf-8"), flags
    )


def h264_full_process(
    frame: np.ndarray,
    *,
    export_mvs: bool = False,
    export_residual: bool = False,
    export_validation: bool = False,
    max_mv: int = 8192,
    max_bytes: int = 2 * 1024 * 1024,
):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a uint8 HWC BGR array.")
    if not frame.flags.c_contiguous:
        raise ValueError("frame must be C-contiguous.")
    if export_validation and not export_residual:
        raise ValueError("Validation planes require residual export.")

    height, width = frame.shape[:2]
    output = (ctypes.c_uint8 * max_bytes)()
    reconstructed = np.empty_like(frame)
    mv_buffer = (MVInfoV2 * max_mv)()
    encoded_size = ctypes.c_int()
    mv_count = ctypes.c_int()
    timing = CodecTiming()

    residual_arrays = None
    residual_storage = None
    residual_output = H264ResidualOutput()
    residual_output_ptr = ctypes.POINTER(H264ResidualOutput)()
    if export_residual:
        chroma_height, chroma_width = height // 2, width // 2
        padded_height = ((height + 15) // 16) * 16
        padded_width = ((width + 15) // 16) * 16
        padded_chroma_height = padded_height // 2
        padded_chroma_width = padded_width // 2
        y = np.empty(padded_height * padded_width, dtype=np.int16)
        u = np.empty(
            padded_chroma_height * padded_chroma_width, dtype=np.int16
        )
        v = np.empty_like(u)
        y_valid = np.empty(y.size, dtype=np.uint8)
        u_valid = np.empty(u.size, dtype=np.uint8)
        v_valid = np.empty_like(u_valid)
        summary_rows = ((width + 15) // 16) * ((height + 15) // 16)
        summaries = np.empty((summary_rows, 10), dtype=np.float32)
        predictions = reconstructions = None
        null_u8 = ctypes.POINTER(ctypes.c_uint8)()
        prediction_ptrs = (null_u8, null_u8, null_u8)
        reconstruction_ptrs = (null_u8, null_u8, null_u8)
        if export_validation:
            predictions = (
                np.empty(y.size, dtype=np.uint8),
                np.empty(u.size, dtype=np.uint8),
                np.empty(u.size, dtype=np.uint8),
            )
            reconstructions = tuple(np.empty_like(item) for item in predictions)
            prediction_ptrs = tuple(
                item.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                for item in predictions
            )
            reconstruction_ptrs = tuple(
                item.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                for item in reconstructions
            )
        residual_output = H264ResidualOutput(
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            y_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            u_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            v_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            *prediction_ptrs,
            *reconstruction_ptrs,
            summaries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.size,
            u.size,
            summary_rows,
            0,
            0,
        )
        residual_output_ptr = ctypes.pointer(residual_output)
        residual_storage = (
            y, u, v, y_valid, u_valid, v_valid,
            predictions, reconstructions, summaries,
        )

    status = lib.h264_full_process(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output, max_bytes, ctypes.byref(encoded_size),
        reconstructed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width * height * 3,
        mv_buffer, max_mv, ctypes.byref(mv_count),
        residual_output_ptr,
        ctypes.byref(timing),
    )
    if status != 0:
        raise RuntimeError(f"H.264 full process failed with status {status}.")

    if residual_storage is not None:
        y, u, v, y_valid, u_valid, v_valid, predictions, reconstructions, summaries = (
            residual_storage
        )
        y_count = height * width
        chroma_count = chroma_height * chroma_width
        if predictions is not None:
            predictions = (
                predictions[0][:y_count].reshape(height, width).copy(),
                predictions[1][:chroma_count]
                .reshape(chroma_height, chroma_width)
                .copy(),
                predictions[2][:chroma_count]
                .reshape(chroma_height, chroma_width)
                .copy(),
            )
            reconstructions = (
                reconstructions[0][:y_count].reshape(height, width).copy(),
                reconstructions[1][:chroma_count]
                .reshape(chroma_height, chroma_width)
                .copy(),
                reconstructions[2][:chroma_count]
                .reshape(chroma_height, chroma_width)
                .copy(),
            )
        residual_arrays = (
            y[:y_count].reshape(height, width).copy(),
            u[:chroma_count].reshape(chroma_height, chroma_width).copy(),
            v[:chroma_count].reshape(chroma_height, chroma_width).copy(),
            y_valid[:y_count].reshape(height, width).copy(),
            u_valid[:chroma_count].reshape(chroma_height, chroma_width).copy(),
            v_valid[:chroma_count].reshape(chroma_height, chroma_width).copy(),
            predictions,
            reconstructions,
            summaries[:residual_output.summary_row_count].copy(),
        )

    vectors = None
    if export_mvs:
        vectors = np.empty((mv_count.value, 9), dtype=np.float32)
        for index in range(mv_count.value):
            item = mv_buffer[index]
            vectors[index] = (
                item.dst_x - item.w / 2.0,
                item.dst_y - item.h / 2.0,
                item.w,
                item.h,
                item.motion_x,
                item.motion_y,
                item.motion_scale,
                item.source,
                item.flags,
            )
    return (
        bytes(output[:encoded_size.value]),
        reconstructed,
        vectors,
        residual_arrays,
        timing,
    )

def mv_close():
    """라이브러리 자원 해제"""
    lib.mv_close()


def h264_full_close():
    lib.h264_full_close()