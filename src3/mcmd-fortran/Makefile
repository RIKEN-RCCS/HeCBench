FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
TARGET = main
SOURCES = force_kernel.f90 main.f90
OBJECTS = force_kernel.o main.o
LAUNCHER ?=
RUN_ARGS ?= mcmd.inp
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY cd ../mcmd-cuda/dataset && $(LAUNCHER) ../../mcmd-fortran/$(TARGET) mcmd.inp && cd ../../mcmd-fortran
clean:
	rm -f $(TARGET) *.o *.mod
