#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /usr/include
KOKKOS_LIB = /usr/lib/x86_64-linux-gnu

CC = g++

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fopenmp -I$(KOKKOS_INC)

LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -fopenmp -lpthread -ldl

ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
