#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

# Path to the installed Kokkos (cmake-based install, e.g. ~/kokkos-install)
KOKKOS_INSTALL   ?= $(HOME)/kokkos-install

# Path to the installed Kokkos Kernels (e.g. ~/kokkos-kernels-install)
KOKKOS_KERNELS_INSTALL ?= $(HOME)/kokkos-kernels-install

# Compiler: use nvcc_wrapper for CUDA backend, g++ for host-only
CC := $(KOKKOS_INSTALL)/bin/nvcc_wrapper

# CUDA architecture (sm_XXX)
CUDA_ARCH ?= sm_121

# cuBLAS / CUDA driver library paths
CUDA_LIB_PATH ?= /usr/local/cuda/lib64

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++20 \
          -I$(KOKKOS_INSTALL)/include \
          -I$(KOKKOS_KERNELS_INSTALL)/include \
          -arch=$(CUDA_ARCH) \
          --expt-extended-lambda \
          -fopenmp

# Linker Flags
LDFLAGS = $(KOKKOS_KERNELS_INSTALL)/lib/libkokkoskernels.a \
          $(KOKKOS_INSTALL)/lib/libkokkoscore.a \
          $(KOKKOS_INSTALL)/lib/libkokkoscontainers.a \
          $(KOKKOS_INSTALL)/lib/libkokkosalgorithms.a \
          $(KOKKOS_INSTALL)/lib/libkokkossimd.a \
          -L$(CUDA_LIB_PATH) -lcublas \
          -L/usr/lib/aarch64-linux-gnu -lcuda \
          -fopenmp -lpthread -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1024 1024 1024 100
