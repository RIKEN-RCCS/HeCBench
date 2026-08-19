FC ?= nvfortran
PROGRAM := main
SOURCES := main.f90
FFLAGS ?= -O3 -mp=gpu -gpu=cc90
LAUNCHER ?=
RUN_ARGS ?= 10

$(PROGRAM): $(SOURCES) Makefile
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod

.PHONY: run clean
