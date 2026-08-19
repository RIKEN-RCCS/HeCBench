FC = nvfortran
FFLAGS = -O3 -Minfo -mp=gpu -gpu=cc90 -Mextend
all: main
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main 100
test: main
	OMP_TARGET_OFFLOAD=MANDATORY ./main 1 32
clean:
	rm -f main *.mod
