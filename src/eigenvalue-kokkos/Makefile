#===============================================================================
# User Options
#===============================================================================
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

program = main
source  = main.cpp \
          ../eigenvalue-omp/reference.cpp
obj     = main.o reference.o

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I../eigenvalue-omp
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

VPATH = ../eigenvalue-omp

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 512 100
