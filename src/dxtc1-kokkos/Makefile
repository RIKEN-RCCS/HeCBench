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
source = main.cpp shrUtils.cpp block.cpp
obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I../dxtc1-cuda

# Linker Flags
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
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

main.o: main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o: ../dxtc1-cuda/shrUtils.cu
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

block.o: ../dxtc1-cuda/block.cu
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) data/lena.ppm data/lena_ref.dds 10
