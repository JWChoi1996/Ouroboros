CC      := gcc
BACKEND := ../codec_backend/build/install
CFLAGS  := -fPIC -O2 -Wall -Wextra -I$(BACKEND)/include
LDFLAGS := -shared \
	-Wl,-Bsymbolic \
	-Wl,-rpath,'$$ORIGIN/../codec_backend/build/install/lib' \
	$(BACKEND)/lib/libavcodec.a \
	$(BACKEND)/lib/libswscale.a \
	$(BACKEND)/lib/libavutil.a \
	$(shell pkg-config --libs x264) -pthread -lm -ldl -lz

all: libcustom_h264.so

libcustom_h264.so: custom_h264.c $(BACKEND)/lib/libavcodec.a
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f libcustom_h264.so
