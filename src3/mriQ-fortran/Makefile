FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
program = main
LAUNCHER ?=
RUN_ARGS ?= ../mriQ-cuda/datasets/128x128x128/input/128x128x128.bin output_128
SOURCES = mriq_kernels.f90 main.f90
OBJECTS = $(SOURCES:.f90=.o)

main: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

../mriQ-cuda/datasets/128x128x128/input/128x128x128.bin: ../mriQ-cuda/mri-q.tgz
	tar -xzf $< --strip-components=1 -C ../mriQ-cuda mri-q/datasets

run: main ../mriQ-cuda/datasets/128x128x128/input/128x128x128.bin
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) ../mriQ-cuda/datasets/128x128x128/input/128x128x128.bin output_128
all: main
clean:
	rm -f main *.o *.mod
