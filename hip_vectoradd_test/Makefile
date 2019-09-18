.PHONY: all clean

HIP_PLATFORM = $(shell hipconfig --platform)

HIPCCFLAGS := -g -O3

ifeq (${HIP_PLATFORM}, nvcc)
	HIPCCFLAGS += --ptxas-options=-v \
		--generate-code arch=compute_60,code=[compute_60,sm_60] \
		--generate-code arch=compute_62,code=[compute_62,sm_62] \
		--generate-code arch=compute_70,code=[compute_70,sm_70] \
		--generate-code arch=compute_72,code=[compute_72,sm_72] \
		--generate-code arch=compute_75,code=[compute_75,sm_75]
endif

all: vector_add

vector_add: vector_add.cpp
	hipcc $(HIPCCFLAGS) -o vector_add vector_add.cpp

clean:
	rm -f vector_add
