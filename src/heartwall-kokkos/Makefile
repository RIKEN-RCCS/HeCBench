#===============================================================================
# User Options
#===============================================================================

CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

OMP_SRC   = ../heartwall-omp

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

# main.cpp (Kokkos port) + AVI/file/timer C utilities from the OMP directory
source    = main.cpp
c_sources = $(OMP_SRC)/util/avi/avilib.c \
            $(OMP_SRC)/util/avi/avimod.c \
            $(OMP_SRC)/util/file/file.c  \
            $(OMP_SRC)/util/timer/timer.c

obj  = $(source:.cpp=.o)
cobj = $(c_sources:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I$(OMP_SRC)/util/avi \
          -I$(OMP_SRC)/util/file \
          -I$(OMP_SRC)/util/timer

CFLAGS_C := -O3 \
            -I$(OMP_SRC) \
            -I$(OMP_SRC)/util/avi \
            -I$(OMP_SRC)/util/file \
            -I$(OMP_SRC)/util/timer

LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -lm

ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS  += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) $(cobj) Makefile
	$(CC) $(CFLAGS) $(obj) $(cobj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Compile the OMP utility C files in-place (they are not modified)
$(OMP_SRC)/util/avi/avilib.o: $(OMP_SRC)/util/avi/avilib.c
	gcc $(CFLAGS_C) -c $< -o $@

$(OMP_SRC)/util/avi/avimod.o: $(OMP_SRC)/util/avi/avimod.c
	gcc $(CFLAGS_C) -c $< -o $@

$(OMP_SRC)/util/file/file.o: $(OMP_SRC)/util/file/file.c
	gcc $(CFLAGS_C) -c $< -o $@

$(OMP_SRC)/util/timer/timer.o: $(OMP_SRC)/util/timer/timer.c
	gcc $(CFLAGS_C) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj) \
	       $(OMP_SRC)/util/avi/avilib.o  \
	       $(OMP_SRC)/util/avi/avimod.o  \
	       $(OMP_SRC)/util/file/file.o   \
	       $(OMP_SRC)/util/timer/timer.o

run: $(program)
	$(LAUNCHER) ./$(program) $(OMP_SRC)/test.avi 20
