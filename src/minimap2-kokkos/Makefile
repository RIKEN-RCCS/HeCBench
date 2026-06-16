OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

CC = g++
program = main

source = device/device_kernel_wrapper.cpp \
         src/host_data_io.cpp \
         src/host_kernel.cpp \
         src/main.cpp \
         src/memory_scheduler.cpp

obj = $(source:.cpp=.o)

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 -I./include -I$(KOKKOS_INC)
LDFLAGS  = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lkokkossimd -lpthread -ldl

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
	rm -rf $(program) $(obj) device/*.o src/*.o

run: $(program)
	$(LAUNCHER) ./$(program) ../minimap2-omp/in-1k.txt kernel-out.txt
