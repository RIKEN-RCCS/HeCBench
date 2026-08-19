FC = nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM := main
SOURCES := main.f90
LAUNCHER ?=
RUN_ARGS ?= 100 0 ../urng-sycl/URNG_Input.bmp
DECRYPT_RUN_ARGS ?= 100 1 ../urng-sycl/URNG_Input.bmp
INPUT_IMAGE := ../urng-sycl/URNG_Input.bmp
INPUT_ARCHIVE := ../urng-sycl/image.tar.gz

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

$(INPUT_IMAGE): $(INPUT_ARCHIVE)
	tar -xzf $< -C $(dir $@)

run: $(PROGRAM) $(INPUT_IMAGE)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(DECRYPT_RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod

.PHONY: run clean
