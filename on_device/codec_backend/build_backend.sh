#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SOURCE_DIR="${BUILD_DIR}/ffmpeg-n8.0.1"
INSTALL_DIR="${BUILD_DIR}/install"
PATCH_FILE="${SCRIPT_DIR}/ffmpeg-8.0.1-ouroboros-residual.patch"
SELFTEST="${BUILD_DIR}/residual_backend_selftest"
MANIFEST="${SCRIPT_DIR}/backend_manifest.json"
FFMPEG_TAG="n8.0.1"
FFMPEG_COMMIT="894da5ca7d742e4429ffb2af534fcda0103ef593"

mkdir -p "${BUILD_DIR}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${FFMPEG_TAG}" \
    https://github.com/FFmpeg/FFmpeg.git "${SOURCE_DIR}"
fi

actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${FFMPEG_COMMIT}" ]]; then
  echo "FFmpeg ${FFMPEG_TAG} resolved to unexpected commit ${actual_commit}" >&2
  exit 1
fi

git -C "${SOURCE_DIR}" reset --hard "${FFMPEG_COMMIT}"
git -C "${SOURCE_DIR}" clean -fdx
git -C "${SOURCE_DIR}" apply --check "${PATCH_FILE}"
git -C "${SOURCE_DIR}" apply "${PATCH_FILE}"
rm -rf "${INSTALL_DIR}"

pushd "${SOURCE_DIR}" >/dev/null
./configure \
  --prefix="${INSTALL_DIR}" \
  --disable-everything \
  --disable-programs --disable-doc --disable-network --disable-autodetect \
  --disable-x86asm --enable-gpl --enable-libx264 \
  --enable-avcodec --enable-avutil --enable-swscale \
  --enable-decoder=h264,mpeg1video --enable-encoder=libx264 \
  --enable-parser=h264 --enable-pic --enable-static --disable-shared
if ! grep -q '^#define CONFIG_MPEGVIDEODEC 1$' config.h; then
  echo "CONFIG_MPEGVIDEODEC is required for H.264 motion-vector export" >&2
  exit 1
fi
make -j"${BUILD_JOBS:-8}"
make install
popd >/dev/null

cc -O2 -std=c11 -Wall -Wextra \
  -I"${INSTALL_DIR}/include" \
  "${SCRIPT_DIR}/residual_backend_selftest.c" \
  -o "${SELFTEST}"
"${SELFTEST}"

patch_sha256="$(sha256sum "${PATCH_FILE}" | cut -d' ' -f1)"
compiler="$(cc --version | sed -n '1p')"
x264_version="$(pkg-config --modversion x264)"
cat >"${MANIFEST}" <<EOF
{
  "ffmpeg_tag": "${FFMPEG_TAG}",
  "ffmpeg_commit": "${FFMPEG_COMMIT}",
  "patch_sha256": "${patch_sha256}",
  "compiler": "${compiler}",
  "libx264_version": "${x264_version}",
  "configure_flags": "--disable-everything --disable-programs --disable-doc --disable-network --disable-autodetect --disable-x86asm --enable-gpl --enable-libx264 --enable-avcodec --enable-avutil --enable-swscale --enable-decoder=h264,mpeg1video --enable-encoder=libx264 --enable-parser=h264 --enable-pic --enable-static --disable-shared"
}
EOF

echo "Built patched FFmpeg ${FFMPEG_TAG} in ${INSTALL_DIR}"
