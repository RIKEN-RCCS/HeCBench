#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /usr/include
KOKKOS_LIB = /usr/lib/x86_64-linux-gnu

CC      = g++
program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fopenmp -I$(KOKKOS_INC)
LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -fopenmp -lpthread -ldl

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
	$(LAUNCHER) ./$(program) 100000 100 10
