FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
program = main
LAUNCHER ?=
RUN_ARGS ?= 128 48 224 224 54 54 100
SOURCES = main.f90
OBJECTS = $(SOURCES:.f90=.o)

main: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) 128 48 224 224 54 54 100
all: main
clean:
	rm -f main *.o *.mod
