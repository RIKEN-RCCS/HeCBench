FC ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend -Minfo
LAUNCHER ?=
RUN_ARGS ?= 64 5
.PHONY: all clean run
all: main
main: main.f90 contract_cases.inc
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ main.f90
run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
