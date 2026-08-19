FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM := main
SOURCES := sobel_mod.f90 bmp_mod.f90 main.f90
LAUNCHER ?=

.PHONY: all run clean

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

../sobel-sycl/SobelFilter_Input.bmp: ../sobel-sycl/data.tar.gz
	tar -xzf $< -C ../sobel-sycl/

run: $(PROGRAM) ../sobel-sycl/SobelFilter_Input.bmp
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) ../sobel-sycl/SobelFilter_Input.bmp 100000
clean:
	rm -f $(PROGRAM) *.o *.mod
