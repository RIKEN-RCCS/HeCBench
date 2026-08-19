FC       = nvfortran
FFLAGS   ?= -O3 -mp=gpu -gpu=cc90 -Mextend
PROGRAM  := main
SOURCES  := main.f90
LAUNCHER ?=
RUN_ARGS ?= 100

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod

.PHONY: run clean
