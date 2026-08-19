FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
INPUT ?= ../../src/inversek2j-omp/coord_in.txt
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main $(INPUT) 100000
clean:
	rm -f main *.mod
