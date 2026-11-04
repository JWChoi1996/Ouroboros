// custom_h264.c  ─  한 프레임 H.264 인코드→디코드 후 모션-벡터 추출
#include <stdint.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/motion_vector.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int16_t dst_x, dst_y;        // 블록 좌표 (좌상단)
    int32_t motion_x, motion_y;  // quarter-pel
    int32_t motion_scale;        // 보통 1 or 2
} MVInfo;

typedef struct {
    int16_t dst_x, dst_y;
    uint8_t w, h;
    int32_t motion_x, motion_y, motion_scale;
    int32_t source;
    uint64_t flags;
} MVInfoV2;

typedef struct {
    uint64_t input_conversion_ns;
    uint64_t encode_ns;
    uint64_t decode_ns;
    uint64_t rgb_materialization_ns;
    uint64_t mv_export_ns;
    uint64_t residual_export_ns;
} CodecTiming;

enum H264ExportFlags {
    H264_EXPORT_NONE = 0,
    H264_EXPORT_MVS = 1 << 0,
    H264_EXPORT_RESIDUAL = 1 << 1,
    H264_EXPORT_VALIDATION = 1 << 2,
};

#define OUROBOROS_RESIDUAL_VERSION 2U
#define OUROBOROS_RESIDUAL_COLUMNS 10U
#define OUROBOROS_RESIDUAL_FLAG_VALIDATION (1U << 0)

typedef struct {
    uint8_t uuid[16];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t chroma_width;
    uint32_t chroma_height;
    uint32_t mb_width;
    uint32_t mb_height;
    uint32_t flags;
    uint32_t payload_bytes;
} OuroborosResidualHeader;

typedef struct {
    int16_t *y;
    int16_t *u;
    int16_t *v;
    uint8_t *y_valid;
    uint8_t *u_valid;
    uint8_t *v_valid;
    uint8_t *prediction_y;
    uint8_t *prediction_u;
    uint8_t *prediction_v;
    uint8_t *reconstruction_y;
    uint8_t *reconstruction_u;
    uint8_t *reconstruction_v;
    float *coefficient_summaries;
    int y_capacity;
    int chroma_capacity;
    int summary_row_capacity;
    int summary_row_count;
    int has_validation;
} H264ResidualOutput;

static const uint8_t OUROBOROS_RESIDUAL_UUID[16] = {
    'O', 'U', 'R', 'O', 'B', 'O', 'R', 'O',
    'S', 'H', '2', '6', '4', 'R', '2', '\0'
};

static AVCodecContext *enc = NULL, *dec = NULL;
static AVFrame *enc_fr = NULL, *dec_fr = NULL;
static struct SwsContext *sws = NULL, *sws_dec = NULL;
static AVPacket *pkt = NULL;
static int pts = 0;
static int active_export_flags = H264_EXPORT_MVS;
static uint8_t *last_residual_payload = NULL;
static size_t last_residual_payload_size = 0;

void mv_close(void);

// 원본 코드에서 사용한 h264 파라미터를 default로 사용
static const char *DEFAULT_X264_PARAMS =
    "keyint=9999:min-keyint=9999:no-scenecut=1:bframes=0:"
    "ref=1:partitions=i16x16,p16x16:8x8dct=0:rc-lookahead=0";

static int init_codec(int w, int h, int fps, const char* x264_params,
                      int requested_export_flags)
{
    const AVCodec *enc_c = avcodec_find_encoder_by_name("libx264");
    const AVCodec *dec_c = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!enc_c || !dec_c) return -1;

    enc = avcodec_alloc_context3(enc_c);
    dec = avcodec_alloc_context3(dec_c);
    if (!enc || !dec) return -1;

    enc->width = w;  enc->height = h;
    enc->pix_fmt = AV_PIX_FMT_YUV420P;
    enc->time_base = (AVRational){1,fps};
    enc->gop_size = 9999; enc->max_b_frames = 0;
    av_opt_set(enc->priv_data,"preset","fast",0);
    av_opt_set(enc->priv_data,"profile","baseline",0);
    av_opt_set(enc->priv_data,"tune","zerolatency",0);
    if (x264_params && strlen(x264_params) > 0)
        av_opt_set(enc->priv_data,"x264-params", x264_params, 0);
    else
        av_opt_set(enc->priv_data,"x264-params", DEFAULT_X264_PARAMS, 0);

    if (avcodec_open2(enc,enc_c,NULL)<0) return -1;

    active_export_flags = requested_export_flags;
    if (active_export_flags & H264_EXPORT_MVS)
        dec->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    if (active_export_flags & H264_EXPORT_RESIDUAL)
        dec->flags2 |= AV_CODEC_FLAG2_EXPORT_OUROBOROS_RESIDUAL;
    if (active_export_flags & H264_EXPORT_VALIDATION)
        dec->flags2 |= AV_CODEC_FLAG2_EXPORT_OUROBOROS_VALIDATION;
    dec->thread_count = 1;
    if (avcodec_open2(dec,dec_c,NULL)<0) return -1;

    enc_fr=av_frame_alloc(); dec_fr=av_frame_alloc(); pkt=av_packet_alloc();
    if(!enc_fr||!dec_fr||!pkt) return -1;
    enc_fr->format=AV_PIX_FMT_YUV420P; enc_fr->width=w; enc_fr->height=h;
    if(av_frame_get_buffer(enc_fr,32)<0) return -1;

    sws = sws_getContext(w,h,AV_PIX_FMT_BGR24,w,h,AV_PIX_FMT_YUV420P,
                         SWS_BILINEAR,NULL,NULL,NULL);
    pts = 0;
    return sws?0:-1;
}

int mv_init(int w, int h, int fps, const char* x264_params)
{
    return init_codec(w, h, fps, x264_params, 1);
}

int codec_init(int w, int h, int fps, const char* x264_params,
               int should_export_mvs)
{
    return init_codec(w, h, fps, x264_params,
                      should_export_mvs ? H264_EXPORT_MVS : H264_EXPORT_NONE);
}

int h264_full_init(int w, int h, int fps, const char *x264_params,
                   int export_flags)
{
    if ((export_flags & H264_EXPORT_VALIDATION) &&
        !(export_flags & H264_EXPORT_RESIDUAL))
        return -10;
    return init_codec(w, h, fps, x264_params, export_flags);
}

int mv_process(uint8_t* bgr, MVInfo *out, int max_out)
{
    int w=enc->width,h=enc->height;
    uint8_t* in[1]={bgr}; int ls[1]={3*w};
    sws_scale(sws,(const uint8_t* const*)in,ls,0,h,enc_fr->data,enc_fr->linesize);

    enc_fr->pts=pts++;
    if(avcodec_send_frame(enc,enc_fr)<0) return -1;

    while(avcodec_receive_packet(enc,pkt)==0){
        if(avcodec_send_packet(dec,pkt)<0){ av_packet_unref(pkt); return -1;}
        av_packet_unref(pkt);

        if(avcodec_receive_frame(dec,dec_fr)==0){
            AVFrameSideData *sd=av_frame_get_side_data(dec_fr,
                                        AV_FRAME_DATA_MOTION_VECTORS);
            int cnt=0;
            if(sd){
                const AVMotionVector *mv=(const AVMotionVector*)sd->data;
                int tot=sd->size/sizeof(*mv);
                for(int i=0;i<tot&&cnt<max_out;i++){
                    out[cnt].dst_x     = mv[i].dst_x;
                    out[cnt].dst_y     = mv[i].dst_y;
                    out[cnt].motion_x  = mv[i].motion_x;
                    out[cnt].motion_y  = mv[i].motion_y;
                    out[cnt].motion_scale = mv[i].motion_scale;
                    ++cnt;
                }
            }
            av_frame_unref(dec_fr);
            return cnt;
        }
    }
    return 0;
}

// 새 시그니처: encoded_size_out, mv_cnt_out 포인터 추가
int mv_process_and_encode(uint8_t* bgr,
                          MVInfo *out, int max_out,
                          uint8_t* out_buf, int out_buf_size,
                          int *encoded_size_out, int *mv_cnt_out)
{
    int w=enc->width, h=enc->height;
    uint8_t* in[1]={bgr}; int ls[1]={3*w};
    sws_scale(sws,(const uint8_t* const*)in,ls,0,h,enc_fr->data,enc_fr->linesize);

    enc_fr->pts=pts++;
    if(avcodec_send_frame(enc,enc_fr)<0) return -1;

    int encoded_size = 0;
    int mv_cnt = 0;
    while(avcodec_receive_packet(enc,pkt)==0){
        if(encoded_size + pkt->size <= out_buf_size) {
            memcpy(out_buf + encoded_size, pkt->data, pkt->size); // 이어붙이기
            encoded_size += pkt->size;
        } else {
            av_packet_unref(pkt);
            return -1; // 버퍼 초과
        }
        if(avcodec_send_packet(dec,pkt)<0){ av_packet_unref(pkt); return -1;}
        av_packet_unref(pkt);

        if(avcodec_receive_frame(dec,dec_fr)==0){
            AVFrameSideData *sd=av_frame_get_side_data(dec_fr, AV_FRAME_DATA_MOTION_VECTORS);
            int cnt=0;
            if(sd){
                const AVMotionVector *mv=(const AVMotionVector*)sd->data;
                int tot=sd->size/sizeof(*mv);
                for(int i=0;i<tot&&cnt<max_out;i++){
                    out[cnt].dst_x     = mv[i].dst_x;
                    out[cnt].dst_y     = mv[i].dst_y;
                    out[cnt].motion_x  = mv[i].motion_x;
                    out[cnt].motion_y  = mv[i].motion_y;
                    out[cnt].motion_scale = mv[i].motion_scale;
                    ++cnt;
                }
            }
            av_frame_unref(dec_fr);
            mv_cnt = cnt;
        }
    }
    *encoded_size_out = encoded_size;
    *mv_cnt_out = mv_cnt;
    return 0;
}

static int capture_residual_side_data(const AVFrame *frame,
                                      CodecTiming *timing)
{
    const AVFrameSideData *side;
    const OuroborosResidualHeader *header;
    uint8_t *copy;
    int64_t started;

    free(last_residual_payload);
    last_residual_payload = NULL;
    last_residual_payload_size = 0;
    if (!(active_export_flags & H264_EXPORT_RESIDUAL))
        return 0;

    started = av_gettime_relative();
    side = av_frame_get_side_data(
        frame, AV_FRAME_DATA_OUROBOROS_H264_RESIDUAL);
    if (!side || side->size < sizeof(*header))
        return -11;
    header = (const OuroborosResidualHeader *)side->data;
    if (memcmp(header->uuid, OUROBOROS_RESIDUAL_UUID,
               sizeof(header->uuid)) ||
        header->version != OUROBOROS_RESIDUAL_VERSION ||
        header->payload_bytes != side->size)
        return -12;
    copy = malloc(side->size);
    if (!copy)
        return -13;
    memcpy(copy, side->data, side->size);
    last_residual_payload = copy;
    last_residual_payload_size = side->size;
    timing->residual_export_ns +=
        (uint64_t)(av_gettime_relative() - started) * 1000;
    return 0;
}

int codec_process(uint8_t *bgr,
                  uint8_t *out_buf, int out_buf_size, int *encoded_size_out,
                  uint8_t *reconstructed_bgr, int reconstructed_size,
                  MVInfoV2 *mv_out, int max_mv, int *mv_cnt_out,
                  CodecTiming *timing_out)
{
    int w, h, encoded_size = 0, mv_count = 0, decoded_frame = 0;
    uint8_t *in[1];
    int in_linesize[1];
    int64_t started;

    if (!enc || !dec || !enc_fr || !dec_fr || !pkt || !sws ||
        !bgr || !out_buf || !encoded_size_out || !reconstructed_bgr ||
        !mv_cnt_out || !timing_out)
        return -1;

    w = enc->width;
    h = enc->height;
    if (reconstructed_size < w * h * 3)
        return -2;

    memset(timing_out, 0, sizeof(*timing_out));
    in[0] = bgr;
    in_linesize[0] = 3 * w;
    started = av_gettime_relative();
    sws_scale(sws, (const uint8_t *const *)in, in_linesize, 0, h,
              enc_fr->data, enc_fr->linesize);
    timing_out->input_conversion_ns =
        (uint64_t)(av_gettime_relative() - started) * 1000;

    enc_fr->pts = pts++;
    started = av_gettime_relative();
    if (avcodec_send_frame(enc, enc_fr) < 0)
        return -3;
    timing_out->encode_ns +=
        (uint64_t)(av_gettime_relative() - started) * 1000;

    for (;;) {
        int receive_status;
        started = av_gettime_relative();
        receive_status = avcodec_receive_packet(enc, pkt);
        timing_out->encode_ns +=
            (uint64_t)(av_gettime_relative() - started) * 1000;
        if (receive_status == AVERROR(EAGAIN) || receive_status == AVERROR_EOF)
            break;
        if (receive_status < 0)
            return -4;
        if (encoded_size + pkt->size > out_buf_size) {
            av_packet_unref(pkt);
            return -5;
        }
        memcpy(out_buf + encoded_size, pkt->data, pkt->size);
        encoded_size += pkt->size;

        started = av_gettime_relative();
        if (avcodec_send_packet(dec, pkt) < 0) {
            av_packet_unref(pkt);
            return -6;
        }
        av_packet_unref(pkt);
        if (avcodec_receive_frame(dec, dec_fr) == 0) {
            timing_out->decode_ns +=
                (uint64_t)(av_gettime_relative() - started) * 1000;
            decoded_frame = 1;

            started = av_gettime_relative();
            sws_dec = sws_getCachedContext(
                sws_dec, w, h, dec_fr->format,
                w, h, AV_PIX_FMT_BGR24, SWS_BILINEAR,
                NULL, NULL, NULL);
            if (!sws_dec) {
                av_frame_unref(dec_fr);
                return -7;
            }
            {
                uint8_t *dst[1] = { reconstructed_bgr };
                int dst_linesize[1] = { 3 * w };
                sws_scale(sws_dec,
                          (const uint8_t *const *)dec_fr->data,
                          dec_fr->linesize, 0, h, dst, dst_linesize);
            }
            timing_out->rgb_materialization_ns =
                (uint64_t)(av_gettime_relative() - started) * 1000;

            if ((active_export_flags & H264_EXPORT_MVS) &&
                mv_out && max_mv > 0) {
                AVFrameSideData *side_data;
                started = av_gettime_relative();
                side_data = av_frame_get_side_data(
                    dec_fr, AV_FRAME_DATA_MOTION_VECTORS);
                if (side_data) {
                    const AVMotionVector *vectors =
                        (const AVMotionVector *)side_data->data;
                    int total = side_data->size / sizeof(*vectors);
                    int i;
                    for (i = 0; i < total && mv_count < max_mv; i++) {
                        mv_out[mv_count].dst_x = vectors[i].dst_x;
                        mv_out[mv_count].dst_y = vectors[i].dst_y;
                        mv_out[mv_count].w = vectors[i].w;
                        mv_out[mv_count].h = vectors[i].h;
                        mv_out[mv_count].motion_x = vectors[i].motion_x;
                        mv_out[mv_count].motion_y = vectors[i].motion_y;
                        mv_out[mv_count].motion_scale =
                            vectors[i].motion_scale;
                        mv_out[mv_count].source = vectors[i].source;
                        mv_out[mv_count].flags = vectors[i].flags;
                        mv_count++;
                    }
                }
                timing_out->mv_export_ns +=
                    (uint64_t)(av_gettime_relative() - started) * 1000;
            }
            {
                int residual_status =
                    capture_residual_side_data(dec_fr, timing_out);
                if (residual_status < 0) {
                    av_frame_unref(dec_fr);
                    return residual_status;
                }
            }
            av_frame_unref(dec_fr);
        } else {
            timing_out->decode_ns +=
                (uint64_t)(av_gettime_relative() - started) * 1000;
        }
    }

    *encoded_size_out = encoded_size;
    *mv_cnt_out = mv_count;
    return decoded_frame ? 0 : -8;
}

static int copy_residual_output(H264ResidualOutput *output,
                                CodecTiming *timing)
{
    const OuroborosResidualHeader *header;
    const int16_t *residual_y, *residual_u, *residual_v;
    const uint8_t *valid_y, *valid_u, *valid_v;
    const uint8_t *cursor;
    const float *summaries;
    size_t y_count, c_count, sample_count, summary_rows, expected_size;
    int64_t started;

    if (!(active_export_flags & H264_EXPORT_RESIDUAL)) {
        if (output) {
            output->summary_row_count = 0;
            output->has_validation = 0;
        }
        return 0;
    }
    if (!output || !last_residual_payload ||
        last_residual_payload_size < sizeof(*header))
        return -14;

    started = av_gettime_relative();
    header = (const OuroborosResidualHeader *)last_residual_payload;
    y_count = (size_t)header->width * header->height;
    c_count = (size_t)header->chroma_width * header->chroma_height;
    sample_count = y_count + 2 * c_count;
    summary_rows = (size_t)header->mb_width * header->mb_height;
    expected_size = sizeof(*header) + sample_count * sizeof(int16_t) +
                    sample_count * sizeof(uint8_t);
    if (header->flags & OUROBOROS_RESIDUAL_FLAG_VALIDATION)
        expected_size += 2 * sample_count * sizeof(uint8_t);
    expected_size =
        (expected_size + sizeof(float) - 1) & ~(sizeof(float) - 1);
    expected_size +=
        summary_rows * OUROBOROS_RESIDUAL_COLUMNS * sizeof(float);
    if (expected_size != last_residual_payload_size)
        return -15;
    if (output->y_capacity < (int)y_count ||
        output->chroma_capacity < (int)c_count ||
        output->summary_row_capacity < (int)summary_rows)
        return -16;
    if (!output->y || !output->u || !output->v ||
        !output->y_valid || !output->u_valid || !output->v_valid ||
        !output->coefficient_summaries)
        return -17;

    residual_y = (const int16_t *)(header + 1);
    residual_u = residual_y + y_count;
    residual_v = residual_u + c_count;
    valid_y = (const uint8_t *)(residual_v + c_count);
    valid_u = valid_y + y_count;
    valid_v = valid_u + c_count;
    cursor = valid_v + c_count;
    memcpy(output->y, residual_y, y_count * sizeof(*output->y));
    memcpy(output->u, residual_u, c_count * sizeof(*output->u));
    memcpy(output->v, residual_v, c_count * sizeof(*output->v));
    memcpy(output->y_valid, valid_y, y_count);
    memcpy(output->u_valid, valid_u, c_count);
    memcpy(output->v_valid, valid_v, c_count);

    output->has_validation = 0;
    if (header->flags & OUROBOROS_RESIDUAL_FLAG_VALIDATION) {
        const uint8_t *prediction_y = cursor;
        const uint8_t *prediction_u = prediction_y + y_count;
        const uint8_t *prediction_v = prediction_u + c_count;
        const uint8_t *reconstruction_y = prediction_v + c_count;
        const uint8_t *reconstruction_u = reconstruction_y + y_count;
        const uint8_t *reconstruction_v = reconstruction_u + c_count;
        if (!output->prediction_y || !output->prediction_u ||
            !output->prediction_v || !output->reconstruction_y ||
            !output->reconstruction_u || !output->reconstruction_v)
            return -18;
        memcpy(output->prediction_y, prediction_y, y_count);
        memcpy(output->prediction_u, prediction_u, c_count);
        memcpy(output->prediction_v, prediction_v, c_count);
        memcpy(output->reconstruction_y, reconstruction_y, y_count);
        memcpy(output->reconstruction_u, reconstruction_u, c_count);
        memcpy(output->reconstruction_v, reconstruction_v, c_count);
        cursor = reconstruction_v + c_count;
        output->has_validation = 1;
    }
    {
        size_t offset = (size_t)(cursor - (const uint8_t *)header);
        offset = (offset + sizeof(float) - 1) & ~(sizeof(float) - 1);
        cursor = (const uint8_t *)header + offset;
    }
    summaries = (const float *)cursor;
    memcpy(output->coefficient_summaries, summaries,
           summary_rows * OUROBOROS_RESIDUAL_COLUMNS * sizeof(float));
    output->summary_row_count = summary_rows;
    timing->residual_export_ns +=
        (uint64_t)(av_gettime_relative() - started) * 1000;
    return 0;
}

int h264_full_process(
    uint8_t *bgr,
    uint8_t *out_buf, int out_buf_size, int *encoded_size_out,
    uint8_t *reconstructed_bgr, int reconstructed_size,
    MVInfoV2 *mv_out, int max_mv, int *mv_cnt_out,
    H264ResidualOutput *residual_output,
    CodecTiming *timing_out)
{
    int status = codec_process(
        bgr, out_buf, out_buf_size, encoded_size_out,
        reconstructed_bgr, reconstructed_size,
        mv_out, max_mv, mv_cnt_out, timing_out);
    if (status < 0)
        return status;
    return copy_residual_output(residual_output, timing_out);
}

void h264_full_close(void)
{
    mv_close();
}

void mv_close(void){
    avcodec_free_context(&enc); avcodec_free_context(&dec);
    av_frame_free(&enc_fr); av_frame_free(&dec_fr);
    av_packet_free(&pkt);
    if(sws) sws_freeContext(sws);
    if(sws_dec) sws_freeContext(sws_dec);
    sws = NULL;
    sws_dec = NULL;
    free(last_residual_payload);
    last_residual_payload = NULL;
    last_residual_payload_size = 0;
    active_export_flags = H264_EXPORT_MVS;
    pts = 0;
}
