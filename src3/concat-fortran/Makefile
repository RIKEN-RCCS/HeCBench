FC=nvfortran
LAUNCHER?=
RUN_ARGS?=1000
FFLAGS?=-O3 -mp=gpu -gpu=cc90 -Mextend
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
.PHONY: run clean
