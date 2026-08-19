FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
RUN_ARGS ?= 8192 8192 100

main: main.f90
	$(FC) $(FFLAGS) -o $@ $<

.PHONY: clean run

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main *.o *.mod
