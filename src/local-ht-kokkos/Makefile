OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

CC        = g++
program   = main
source    = main.cpp
obj       = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(KOKKOS_INC)
LDFLAGS  = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lpthread -ldl

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
	rm -rf $(program) $(obj) out_file.dat

run: $(program)
	$(LAUNCHER) ./$(program) ../local-ht-cuda/locassm_data/localassm_extend_7-21.dat 21 ./out_file.dat
