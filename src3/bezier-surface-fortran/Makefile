FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
RUN_ARGS ?= -n 8192
INPUT_FILE ?= ../../src/bezier-surface-omp/input/control.txt
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main -f $(INPUT_FILE) $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
