FC = nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
RUN_ARGS ?= 10000000 1000

main: main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main *.mod
