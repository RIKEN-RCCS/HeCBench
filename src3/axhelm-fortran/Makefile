FC          = nvfortran
OPTFLAGS    ?= -O3
GPU_ARCH    ?= cc90
FFLAGS      ?= $(OPTFLAGS) -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LDFLAGS     ?= -lgfortran
PROGRAM     := main
SOURCES     := mesh_basis.f90 main.f90
LAUNCHER    ?=
RUN_ARGS    ?= 1 8000 100
RUN_ARGS_2  ?= 3 8000 100
BLAS_ARCHIVE ?= ../axhelm-omp/BlasLapack.tar.gz
BLAS_DIR    := BlasLapack
BLAS_LIB    := $(BLAS_DIR)/libBlasLapack.a

.PHONY: all clean run

all: $(PROGRAM)

$(BLAS_LIB): $(BLAS_ARCHIVE)
	tar -xzf $(BLAS_ARCHIVE)
	$(MAKE) -C $(BLAS_DIR) lib

$(PROGRAM): $(SOURCES) $(BLAS_LIB)
	$(FC) $(FFLAGS) -o $@ $(SOURCES) $(BLAS_LIB) $(LDFLAGS)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS_2)

clean:
	rm -f $(PROGRAM) *.o *.mod
	if [ -d $(BLAS_DIR) ]; then $(MAKE) -C $(BLAS_DIR) clean; rmdir $(BLAS_DIR) 2>/dev/null || true; fi
