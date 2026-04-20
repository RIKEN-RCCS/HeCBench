#===============================================================================
# User Options
#===============================================================================

CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp kernels.cpp utils.cpp

obj = main.o kernels.o utils.o

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -I../simpleSpmv-cuda

# Linker Flags
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
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

kernels.o: kernels.cpp ../simpleSpmv-cuda/mv.h
$(CC) $(CFLAGS) -c $< -o $@

utils.o: ../simpleSpmv-cuda/utils.cpp ../simpleSpmv-cuda/mv.h
$(CC) $(CFLAGS) -c $< -o $@

main.o: ../simpleSpmv-cuda/main.cpp ../simpleSpmv-cuda/mv.h
$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
rm -rf $(program) $(obj)

run: $(program)
$(LAUNCHER) ./$(program) 16777216 10240 100
