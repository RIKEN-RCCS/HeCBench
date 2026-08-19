FC      ?= nvfortran
FFLAGS  ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
INPUT   ?= ../geodesic-omp/locations.txt

$(PROGRAM): distance.f90 main.f90
	$(FC) $(FFLAGS) -o $@ $^

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) $(INPUT) 100

clean:
	rm -f $(PROGRAM) *.mod
