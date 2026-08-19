FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90
LAUNCHER ?=
RUN_ARGS ?= 25 2
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
