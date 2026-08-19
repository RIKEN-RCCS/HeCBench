FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= 32 100
PROGRAM := main
SOURCES := reference.f90 main.f90

.PHONY: all clean run
all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) 32 100
clean:
	rm -f $(PROGRAM) *.o *.mod
