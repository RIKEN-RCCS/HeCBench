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
sources = main.cpp \
          vector.cpp \
          hamiltonian.cpp \
          sigma.cpp \
          lsqt.cpp \
          model.cpp \
          model_general.cpp \
          model_lattice.cpp \
          anderson.cpp \
          charge.cpp
objs    = $(sources:.cpp=.o)

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

$(program): $(objs) Makefile
	$(CC) $(CFLAGS) $(objs) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(objs)

run: $(program)
	$(LAUNCHER) ./$(program) ../lsqt-omp/test/
