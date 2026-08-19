FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main 128 128 128 100 1
	OMP_TARGET_OFFLOAD=MANDATORY ./main 512 512 512 100 0
clean:
	rm -f main *.o *.mod
