FC       ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend -Minfo
LAUNCHER ?=
RUN_ARGS ?= 160 400 100

program = main
source = main.f90

.PHONY: all clean run

all: $(program)

$(program): $(source)
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $(source)

run: $(program)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) $(RUN_ARGS)

clean:
	rm -f $(program) *.o *.mod
