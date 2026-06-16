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

cpp_source = main.cpp
c_source   = fbase.c fparam.c util.c

obj = $(cpp_source:.cpp=.o) $(c_source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -DINT $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)

# C files need a compatible set of flags (no C++-only options)
CFLAGS_C := -DINT -std=c11 -O3

LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -lm

ifeq ($(DEBUG),yes)
  CFLAGS   += -g -DDEBUG
  CFLAGS_C += -g
  LDFLAGS  += -g
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

%.o: %.c
	$(CC) $(CFLAGS_C) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../frna-omp/LRPPRC_HUMAN.fasta frna_output.txt
