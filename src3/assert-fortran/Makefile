FC = nvfortran
ARCH ?= cc90
OPTFLAGS ?= -O3
FFLAGS ?= $(OPTFLAGS) -mp=gpu -gpu=$(ARCH)
PROGRAM := main
SOURCES := main.f90
LAUNCHER ?=
RUN_ARGS ?=

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
