FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM:=main
LAUNCHER?=
RUN_ARGS?=10000
all: $(PROGRAM)
$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
clean:
	rm -f $(PROGRAM) *.o *.mod
