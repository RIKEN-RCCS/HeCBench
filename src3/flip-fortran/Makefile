FC      ?= nvfortran
FFLAGS  ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main

$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) 3 1024 100

clean:
	rm -f $(PROGRAM) *.mod
