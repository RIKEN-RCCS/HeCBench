FC?=nvfortran
GPU_ARCH?=cc90
LAUNCHER ?=
LOCATIONS := ../geodesic-omp/locations.txt
LOCATIONS_ARCHIVE := ../geodesic-sycl/locations.tar.gz
all: main
main: main.f90
	$(FC) -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend -o $@ $<
$(LOCATIONS): $(LOCATIONS_ARCHIVE)
	tar -xzf $< -C $(dir $@)

run: main $(LOCATIONS)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(if $(RUN_ARGS),$(RUN_ARGS),1000)
clean:
	rm -f main *.o *.mod
