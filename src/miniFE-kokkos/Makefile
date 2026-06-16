OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

KOKKOS_INC = /home/copilot/kokkos-install/include
KOKKOS_LIB = /home/copilot/kokkos-install/lib

CC = g++
program = miniFE.x

MINIFE_TYPES = \
    -DMINIFE_SCALAR=double \
    -DMINIFE_LOCAL_ORDINAL=int \
    -DMINIFE_GLOBAL_ORDINAL=int \
    -DMINIFE_RESTRICT=__restrict__

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

MINIFE_INFO = 1
MINIFE_KERNELS = 0

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 \
    -I$(KOKKOS_INC) \
    -Isrc -Iutils -Ifem \
    $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) \
    -DMINIFE_INFO=$(MINIFE_INFO) -DMINIFE_KERNELS=$(MINIFE_KERNELS)

LDFLAGS = -L$(KOKKOS_LIB) -lkokkoscore -lkokkoscontainers -lpthread -ldl

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

SOURCES = src/main.cpp src/YAML_Doc.cpp src/YAML_Element.cpp \
          utils/BoxPartition.cpp utils/mytimer.cpp utils/param_utils.cpp utils/utils.cpp
OBJS = $(SOURCES:.cpp=.o)

$(program): generate_info $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

generate_info:
	cd src && ./generate_info_header "$(CC)" "$(CFLAGS)" "miniFE" "MINIFE"

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(OBJS) src/miniFE_info.hpp

run: $(program)
	$(LAUNCHER) ./$(program) nx=64 ny=64 nz=64
