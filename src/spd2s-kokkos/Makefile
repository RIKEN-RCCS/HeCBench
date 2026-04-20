KOKKOS_INC = $(HOME)/kokkos-install/include
KOKKOS_LIB = $(HOME)/kokkos-install/lib
CC = g++
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -O3 -I$(KOKKOS_INC)
LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lpthread -ldl
program = main
source = main.cpp
obj = $(source:.cpp=.o)
$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf $(program) $(obj)
run: $(program)
	$(LAUNCHER) ./$(program)
