FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
program = main
LAUNCHER ?=
RUN_ARGS ?= 640 480
SOURCES = overlay_reference.f90 main.f90
OBJECTS = $(SOURCES:.f90=.o)

main: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) 640 480
all: main
clean:
	rm -f main *.o *.mod
