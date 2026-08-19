FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main 2048 2048 1000
clean:
	rm -f main *.mod *.bin
