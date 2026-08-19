FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
RUN_ARGS ?= 65536
LAUNCHER ?=

$(PROGRAM): main.f90 Makefile
	$(FC) $(FFLAGS) -o $@ main.f90

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
