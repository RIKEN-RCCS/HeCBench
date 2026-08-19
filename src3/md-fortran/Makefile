FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
TARGET = main
SOURCES = main.f90
OBJECTS = main.o
LAUNCHER ?=
RUN_ARGS ?= 3 1000
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) 3 1000
clean:
	rm -f $(TARGET) *.o *.mod
