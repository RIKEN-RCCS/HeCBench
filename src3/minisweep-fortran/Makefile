FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
TARGET = main
SOURCES = utils.f90 kernels.f90 main.f90
OBJECTS = utils.o kernels.o main.o
LAUNCHER ?=
RUN_ARGS ?= --niterations 100
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) --niterations 100
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) --niterations 1 --ncell_x 32 --ncell_y 32 --ncell_z 64 --ne 64 --na 32 --nblock_z 64
clean:
	rm -f $(TARGET) *.o *.mod
