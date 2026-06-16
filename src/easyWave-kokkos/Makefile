#===============================================================================
# User Options
#===============================================================================
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

program = main
source  = main.cpp \
          ../easyWave-omp/cOgrd.cpp ../easyWave-omp/cOkadaEarthquake.cpp \
          ../easyWave-omp/cOkadaFault.cpp ../easyWave-omp/cSphere.cpp \
          ../easyWave-omp/utilits.cpp ../easyWave-omp/okada.cpp
obj     = $(patsubst %.cpp,%.o,$(notdir $(source)))

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I../easyWave-omp
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

VPATH = ../easyWave-omp

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -grid ../easyWave-omp/data/e2Asean.grd -source ../easyWave-omp/data/aus1.flt -time 20
