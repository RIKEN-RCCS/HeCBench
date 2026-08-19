FC ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend -Minfo
LAUNCHER ?=
RUN_ARGS ?= 10
DATA_DIR ?= ../clink-omp
DATA_ARCHIVE := $(DATA_DIR)/data.tar.gz
DATA_FILES := $(DATA_DIR)/input.hpp $(DATA_DIR)/weight_1.hpp $(DATA_DIR)/weight_2.hpp
DATA_READY := $(DATA_DIR)/.fortran-data-ready
.PHONY: all clean run
all: main
main: main.f90
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $<

$(DATA_READY): $(DATA_ARCHIVE)
	tar -xzf $< -C $(DATA_DIR)
	touch $@

$(DATA_FILES): $(DATA_READY)

run: main $(DATA_FILES)
	CLINK_DATA_DIR=$(DATA_DIR) OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
