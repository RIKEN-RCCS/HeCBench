#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

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

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(KOKKOS_INC)

LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lpthread -ldl

ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
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

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 0 1
