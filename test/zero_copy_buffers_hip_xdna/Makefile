DRIVER_PATH?=../..
CXX=g++
CXX=g++-13
HIPCC=hipcc
INCLUDES=-I$(DRIVER_PATH)/src/include/uapi/drm_local 
LDFLAGS= 
LIBS= -I/usr/include/libdrm -lhsa-runtime64 -ldrm
CFLAGS=-O1 -g
SRC=zero_copy_buffer.cpp cmd.h ipu-util.h

all: zero_copy_buffer.elf

zero_copy_buffer.elf: zero_copy_buffer.cpp
	$(HIPCC) -std=c++17 $(CFLAGS) $(INCLUDES) $(LIBS) $(LDFLAGS) zero_copy_buffer.cpp -o zero_copy_buffer.elf

clean:
	rm -f zero_copy_buffer.elf

format:
	for file in $(SRC); do \
	    clang-format -style=llvm -i $$file; \
	done


.PHONY: tags
tags:
	ctags -R -f tags .  $(DRIVER_PATH)
