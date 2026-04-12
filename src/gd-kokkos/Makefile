OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_INC = /usr/include
KOKKOS_LIB = /usr/lib/x86_64-linux-gnu
CC = g++
program = main
source  = main.cpp
obj     = main.o utils.o
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fopenmp -I$(KOKKOS_INC) -I../gd-sycl
LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -fopenmp -lpthread -ldl
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)
utils.o: ../gd-sycl/utils.cpp ../gd-sycl/utils.h
	$(CC) $(CFLAGS) -c $< -o $@
main.o: main.cpp ../gd-sycl/utils.h
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf $(program) $(obj)
run: $(program)
	$(LAUNCHER) ./$(program) gisette_scale 0.0001 10 100
