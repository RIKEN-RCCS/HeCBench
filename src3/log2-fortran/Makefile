FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
TARGET = main
SOURCES = log2_kernel.f90 main.f90
OBJECTS = log2_kernel.o main.o
LAUNCHER ?=
RUN_ARGS ?= ../log2-cuda/log2_parameters.config
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) ../log2-cuda/log2_parameters.config
clean:
	rm -f $(TARGET) *.o *.mod
