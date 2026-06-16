OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =
KOKKOS_PATH ?= $(HOME)/kokkos

CC  = $(KOKKOS_CXX)

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)
LDFLAGS = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) $(KOKKOS_LINK_DEPENDS) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
