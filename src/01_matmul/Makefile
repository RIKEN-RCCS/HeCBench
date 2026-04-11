#===============================================================================
# User Options
#===============================================================================

OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

# OpenACC compiler — override with e.g. make CC=nvc++
# nvc++ (NVHPC) or g++ (GCC with -fopenacc) are both supported.
CC        = nvc++

# Target GPU architecture for nvc++ (e.g. cc80, cc86, cc89, cc90)
GPU_ARCH  ?= cc80

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Detect compiler: use -fopenacc for g++, -acc=gpu for nvc++
ifeq ($(findstring nvc,$(CC)),nvc)
  ACC_FLAGS = -acc=gpu -gpu=$(GPU_ARCH) -Minfo=accel
else
  # Assume GCC-compatible (g++, etc.)
  ACC_FLAGS = -fopenacc
endif

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(ACC_FLAGS)

# Linker Flags
LDFLAGS = $(ACC_FLAGS)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
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
	$(CC) $(LDFLAGS) $(obj) -o $@

%.o: %.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1024 1024 1024 100
