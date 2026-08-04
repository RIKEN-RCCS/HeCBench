CC        = $(KOKKOS_CXX)
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

program = main
source = main.cpp srad.cpp
obj = $(source:.cpp=.o)

UTIL_OBJS = graphics.o resize.o timer.o

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          -I. \
          -I./util/graphics \
          -I./util/timer
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -lm

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) $(UTIL_OBJS) Makefile
	$(CC) $(CFLAGS) $(obj) $(UTIL_OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

graphics.o: util/graphics/graphics.c
	$(CC) -O3 -x c -c $< -o $@

resize.o: util/graphics/resize.c
	$(CC) -O3 -x c -c $< -o $@

timer.o: util/timer/timer.c
	$(CC) -O3 -x c -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj) $(UTIL_OBJS)

run: $(program)
	$(LAUNCHER) ./$(program) 1000 0.5 502 458
