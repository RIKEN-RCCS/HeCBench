FC         = nvfortran
OPTFLAGS   ?= -O3
GPU_ARCH   ?= cc90
FFLAGS     ?= $(OPTFLAGS) -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM    := main
SOURCES    := main.f90
LAUNCHER   ?=
RUN_ARGS   ?= 1 10
RUN_ARGS_2 ?= 16 10

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS_2)

clean:
	rm -f $(PROGRAM) *.o *.mod
