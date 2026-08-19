FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
TARGET = lulesh
SOURCES = lulesh_types.f90 lulesh_kernels.f90 main.f90
OBJECTS = lulesh_types.o lulesh_kernels.o main.o
LAUNCHER ?=
RUN_ARGS ?= -i 100 -s 128 -r 11 -b 1 -c 1
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) -i 100 -s 128 -r 11 -b 1 -c 1
clean:
	rm -f $(TARGET) *.o *.mod
