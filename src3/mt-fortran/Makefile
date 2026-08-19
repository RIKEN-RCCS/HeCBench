FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
program = main
LAUNCHER ?=
RUN_ARGS ?= 1000
SOURCES = mt_types.f90 genmtrand.f90 mt_kernel.f90 mt_gold.f90 main.f90
OBJECTS = $(SOURCES:.f90=.o)

main: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) 1000
all: main
clean:
	rm -f main *.o *.mod
