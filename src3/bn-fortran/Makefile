FC ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend
PROGRAM := main
SOURCE := main.f90
LAUNCHER ?=
RUN_ARGS ?= result 10

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCE) data45.dat
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $(SOURCE)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod result
