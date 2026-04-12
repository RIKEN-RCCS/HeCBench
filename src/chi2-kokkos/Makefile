OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_INC = /usr/include
KOKKOS_LIB = /usr/lib/x86_64-linux-gnu
CC = g++
program = main
source  = main.cpp
obj     = $(source:.cpp=.o)
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fopenmp -I$(KOKKOS_INC) -I../chi2-cuda
LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -fopenmp -lpthread -ldl
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
	$(LAUNCHER) ./$(program) 4000 400000 2000 2000 256 1000
