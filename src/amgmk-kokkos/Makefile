#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

#===============================================================================
# Program name & source code list
#===============================================================================

program = AMGMk

source = csr_matrix.cpp csr_matvec.cpp hypre_error.cpp hypre_memory.cpp laplace.cpp main.cpp vector.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)

# Linker Flags
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
