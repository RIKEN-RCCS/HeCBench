#===============================================================================
# User Options
#===============================================================================
CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

# Set HAVE_BOOST=yes and BOOST_ROOT=/path/to/boost to enable CPU verification
HAVE_BOOST ?= no
BOOST_ROOT ?=

#===============================================================================
program = main
source = main.cpp
obj = $(source:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I. -I../interval-omp

LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(HAVE_BOOST),yes)
  CFLAGS  += -DHAVE_BOOST -I$(BOOST_ROOT) -I../interval-cuda
  LDFLAGS +=
endif

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 0 100000
	$(LAUNCHER) ./$(program) 1 100000
