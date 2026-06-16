#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
GSL_INC   = -I/path/to/gsl/include
GSL_LIB   = -L/path/to/gsl/lib -lgsl -lgslcblas
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = basis.o scf.o main.o crys.o cints.o

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC)

# Linker Flags
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) $(GSL_LIB)

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

main.o: main.cpp cuda_rys_sp.cpp cuda_rys_dp.cpp
	$(CC) $(CFLAGS) -c $< -o $@

scf.o: ../xlqc-cuda/scf.cc
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

basis.o: ../xlqc-cuda/basis.cc
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

crys.o: ../xlqc-cuda/int_lib/crys.cc
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

cints.o: ../xlqc-cuda/int_lib/cints.cc
	$(CC) $(CFLAGS) -x c++ -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) sp
	$(LAUNCHER) ./$(program) dp
