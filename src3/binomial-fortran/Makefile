FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90
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
