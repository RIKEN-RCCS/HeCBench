FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main -x 5120 -y 5120 -w 10 -n 1000
clean:
	rm -f main *.mod
