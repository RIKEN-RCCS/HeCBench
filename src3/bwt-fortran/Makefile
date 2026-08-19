FC       ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
RUN_ARGS ?= 10000000

program = main

.PHONY: all clean run
all: $(program)

$(program): main.f90
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $<

run: $(program)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) $(RUN_ARGS)

clean:
	rm -f $(program) *.o *.mod
