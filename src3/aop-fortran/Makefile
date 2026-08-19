FC ?= nvfortran
SM ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(SM) -Mextend -Minfo=mp
PROGRAM := main
SOURCES := main.f90
LAUNCHER ?=
RUN_ARGS ?= -runs 100

$(PROGRAM): $(SOURCES) Makefile
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
