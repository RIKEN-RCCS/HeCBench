FC = nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
INPUT_IMAGE ?= ../../src/boxfilter-sycl/data/lenaRGB.ppm
RUN_ARGS ?= $(INPUT_IMAGE) 10000

main: main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main *.o *.mod
