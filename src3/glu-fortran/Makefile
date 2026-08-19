FC?=nvfortran
GPU_ARCH?=cc90
all: main
main: main.f90
	$(FC) -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend -o $@ $<
run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(if $(RUN_ARGS),$(RUN_ARGS),3 128 100)
clean:
	rm -f main *.o *.mod
