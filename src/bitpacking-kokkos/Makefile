CC          = $(KOKKOS_CXX)
KOKKOS_PATH ?= $(HOME)/kokkos
program     = main
source      = main.cpp kernels.cpp utils.cpp
obj         = $(source:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -O3
LDFLAGS  = $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean: kokkos-clean
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
