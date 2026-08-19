FC       ?= nvfortran
OPTFLAGS ?= -O3
GPUFLAGS ?= -mp=gpu -gpu=cc90 -Mextend
PROGRAM  := main
SOURCE   := main.f90
LAUNCHER ?=
RUN_ARGS ?= 1024

$(PROGRAM): $(SOURCE)
	$(FC) $(OPTFLAGS) $(GPUFLAGS) -o $@ $<

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
