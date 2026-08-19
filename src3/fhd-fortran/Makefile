FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
LAUNCHER ?=
RUN_ARGS ?= 100000 2097152 0

$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

test: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) 1000 8192 1

clean:
	rm -f $(PROGRAM) *.o *.mod
