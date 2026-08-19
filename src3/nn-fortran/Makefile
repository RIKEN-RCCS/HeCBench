FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
program = main
LAUNCHER ?=
RUN_ARGS ?= filelist.txt -r 5 -lat 30 -lng 90 -i 10000 -t
SOURCES = utils.f90 nearest_neighbor.f90
OBJECTS = $(SOURCES:.f90=.o)

main: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) filelist.txt -r 5 -lat 30 -lng 90 -i 10000 -t
all: main
clean:
	rm -f main *.o *.mod
