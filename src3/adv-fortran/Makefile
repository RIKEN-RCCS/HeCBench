FC       ?= nvfortran
SM       ?= cc90
FFLAGS   ?= -O3 -mp=gpu -gpu=$(SM),maxregcount:168 -Minfo=mp
PROGRAM  := main
SOURCES  := main.f90
LAUNCHER ?=
RUN_ARGS ?= 7 15 8000 100
DUMP     ?= no

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCES) Makefile
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	ADV_DUMP=$(DUMP) OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
