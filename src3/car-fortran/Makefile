FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
SOURCE := main.f90
LAUNCHER ?=
RUN_ARGS ?= 100

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCE)
	$(FC) $(FFLAGS) $(SOURCE) -o $@

run: $(PROGRAM)
	env OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
