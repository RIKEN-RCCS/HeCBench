#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INSTALL   ?= $(HOME)/kokkos-install
CUDA_ARCH        ?= sm_121
CUDA_LIB_PATH    ?= /usr/local/cuda/lib64

CC := $(KOKKOS_INSTALL)/bin/nvcc_wrapper

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++20 \
          -I$(KOKKOS_INSTALL)/include \
          -I../bilateral-cuda \
          -arch=$(CUDA_ARCH) \
          --expt-extended-lambda \
          -fopenmp

LDFLAGS = $(KOKKOS_INSTALL)/lib/libkokkoscore.a \
          $(KOKKOS_INSTALL)/lib/libkokkoscontainers.a \
          $(KOKKOS_INSTALL)/lib/libkokkosalgorithms.a \
          $(KOKKOS_INSTALL)/lib/libkokkossimd.a \
          -L$(CUDA_LIB_PATH) -lcudart -lcuda -L/usr/lib/aarch64-linux-gnu \
          -fopenmp -lpthread -ldl

ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../bilateral-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1024 1024 0.1 3.0 100
