FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main 256 16777216 100
clean:
	rm -f main *.o *.mod
