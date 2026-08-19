FC ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
SOURCE := main.f90
LAUNCHER ?=
RUN_ARGS ?=

$(PROGRAM): $(SOURCE) Makefile
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $(SOURCE)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
