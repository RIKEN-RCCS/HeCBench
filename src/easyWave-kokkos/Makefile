#===============================================================================
# User Options
#===============================================================================
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

program = main
source  = main.cpp easyWave.cpp \
          cOgrd.cpp cOkadaEarthquake.cpp \
          cOkadaFault.cpp cSphere.cpp \
          utilits.cpp okada.cpp
obj     = $(patsubst %.cpp,%.o,$(notdir $(source)))

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I.
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

VPATH = .

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -grid data/grids/e2Asean.grd -source data/faults/BengkuluSept2007.flt -time 20
