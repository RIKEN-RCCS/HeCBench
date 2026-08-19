FC = nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
SOURCE := main.f90
LAUNCHER ?=
RUN_ARGS ?= 268435456 10

$(PROGRAM): $(SOURCE)
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod

.PHONY: run clean
