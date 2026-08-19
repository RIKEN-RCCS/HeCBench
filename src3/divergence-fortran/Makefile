FC := nvfortran
FFLAGS := -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := divergence
LAUNCHER ?=
INPUT_FILE ?= ../../src/divergence-omp/input.txt
RUN_ARGS ?= $(INPUT_FILE) 10000

$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM) $(INPUT_FILE)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
