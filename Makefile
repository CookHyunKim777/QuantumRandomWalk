CC ?= cc
CFLAGS ?= -O3 -fPIC
LDLIBS ?= -lm

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
LIBRARY := libqrw.dylib
SHARED_FLAG := -dynamiclib
else
LIBRARY := libqrw.so
SHARED_FLAG := -shared
endif

.PHONY: all verify clean

all: $(LIBRARY)

$(LIBRARY): qrw_core.c qrw_core.h
	$(CC) $(CFLAGS) $(SHARED_FLAG) -o $@ qrw_core.c $(LDLIBS)

verify: $(LIBRARY)
	python3 qrw_c_bridge.py --verify

clean:
	$(RM) libqrw.so libqrw.dylib
