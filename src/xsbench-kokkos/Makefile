#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
PROFILE   = no
MPI       = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

obj = Main.o io.o Simulation.o GridInit.o XSutils.o Materials.o

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -I../xsbench-cuda

# Linker Flags
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
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

Main.o: Main.cpp ../xsbench-cuda/XSbench_header.h
	$(CC) $(CFLAGS) -c $< -o $@

Simulation.o: Simulation.cpp ../xsbench-cuda/XSbench_header.h
	$(CC) $(CFLAGS) -c $< -o $@

io.o: ../xsbench-cuda/io.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

GridInit.o: ../xsbench-cuda/GridInit.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

Materials.o: ../xsbench-cuda/Materials.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

XSutils.o: ../xsbench-cuda/XSutils.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -s large -m event -r 10
