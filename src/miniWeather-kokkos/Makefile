OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

CC = mpicxx
program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 -I$(KOKKOS_INC) \
	-DNX=400 \
	-DNZ=200 \
	-DSIM_TIME=600 \
	-DDATA_SPEC=DATA_SPEC_THERMAL

LDFLAGS  = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lkokkossimd -lpthread -ldl -lmpi

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
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) mpiexec -n 1 ./$(program)
