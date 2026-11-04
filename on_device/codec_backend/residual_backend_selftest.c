#include <assert.h>
#include <stddef.h>
#include <stdint.h>


#define RESIDUAL_VERSION 2U
#define SUMMARY_COLUMNS 10U
#define FLAG_VALIDATION (1U << 0)

typedef struct OuroborosResidualHeader {
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

static size_t payload_size(uint32_t width, uint32_t height, uint32_t flags)
{
    size_t y_count = (size_t)width * height;
    size_t c_count = (size_t)((width + 1) / 2) * ((height + 1) / 2);
    size_t sample_count = y_count + 2 * c_count;
    size_t mb_count = (size_t)((width + 15) / 16) *
                      ((height + 15) / 16);
    size_t offset = sizeof(OuroborosResidualHeader);
    offset += sample_count * sizeof(int16_t);
    offset += sample_count * sizeof(uint8_t);
    if (flags & FLAG_VALIDATION)
        offset += 2 * sample_count * sizeof(uint8_t);
    offset = (offset + _Alignof(float) - 1) &
             ~((size_t)_Alignof(float) - 1);
    offset += mb_count * SUMMARY_COLUMNS * sizeof(float);
    return offset;
}

static size_t summary_offset(uint32_t width, uint32_t height, uint32_t flags)
{
    size_t y_count = (size_t)width * height;
    size_t c_count = (size_t)((width + 1) / 2) * ((height + 1) / 2);
    size_t sample_count = y_count + 2 * c_count;
    size_t offset = sizeof(OuroborosResidualHeader);
    offset += sample_count * sizeof(int16_t);
    offset += sample_count * sizeof(uint8_t);
    if (flags & FLAG_VALIDATION)
        offset += 2 * sample_count * sizeof(uint8_t);
    return (offset + _Alignof(float) - 1) &
           ~((size_t)_Alignof(float) - 1);
}

int main(void)
{
    OuroborosResidualHeader header = {
        .version = RESIDUAL_VERSION,
        .width = 32,
        .height = 32,
        .chroma_width = 16,
        .chroma_height = 16,
        .mb_width = 2,
        .mb_height = 2,
        .flags = FLAG_VALIDATION,
    };
    const size_t y_count = (size_t)header.width * header.height;
    const size_t u_count =
        (size_t)header.chroma_width * header.chroma_height;
    const size_t v_count = u_count;
    const size_t summary_count =
        (size_t)header.mb_width * header.mb_height * SUMMARY_COLUMNS;
    size_t last_offset;

    header.payload_bytes =
        payload_size(header.width, header.height, header.flags);
    last_offset = sizeof(header);
    last_offset += (y_count + u_count + v_count) * sizeof(int16_t);
    last_offset += (y_count + u_count + v_count) * sizeof(uint8_t);
    last_offset += 2 * (y_count + u_count + v_count) * sizeof(uint8_t);
    last_offset = (last_offset + _Alignof(float) - 1) &
                  ~((size_t)_Alignof(float) - 1);
    last_offset += summary_count * sizeof(float);

    assert(header.version == 2);
    assert(y_count == 32 * 32);
    assert(u_count == 16 * 16);
    assert(v_count == 16 * 16);
    assert(summary_count == 2 * 2 * 10);
    assert(last_offset == header.payload_bytes);
    assert(summary_offset(66, 66, FLAG_VALIDATION) % _Alignof(float) == 0);
    return 0;
}
