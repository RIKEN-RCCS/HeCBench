#===============================================================================
# User Options
#===============================================================================
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

CC      = g++
program = main

source  = main.cpp \
          ../cmp-cuda/log.cpp \
          ../cmp-cuda/parser.cpp \
          ../cmp-cuda/reference.cpp \
          ../cmp-cuda/su_cdp.cpp \
          ../cmp-cuda/su_gather.cpp \
          ../cmp-cuda/su_trace.cpp

obj     = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(KOKKOS_INC) -I../cmp-cuda
LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lpthread -ldl

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -c0 1.2 -c1 1.8 -nc 5 -aph 600 -tau 0.002 -i ../cmp-omp/data -v 0
