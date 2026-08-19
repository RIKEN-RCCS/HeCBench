FC = nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
LAUNCHER ?=
RUN_ARGS ?= 10
$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
clean:
	rm -f $(PROGRAM) *.o *.mod
.PHONY: run clean
