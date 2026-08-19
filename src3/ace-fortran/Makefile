FC = nvfortran
OPTIMIZE ?= yes
SM ?= cc90
LAUNCHER ?=
RUN_ARGS ?= 100

FFLAGS = -mp=gpu -gpu=$(SM) -Mextend
ifeq ($(OPTIMIZE),yes)
FFLAGS += -O3
endif

program = main
source = main.f90

$(program): $(source)
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(program) *.mod

run: $(program)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) $(RUN_ARGS)
