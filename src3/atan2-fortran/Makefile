FC = nvfortran
OPTIMIZE ?= yes
DEBUG ?= no
SM ?= cc90
LAUNCHER ?=
RUN_ARGS ?= 10000000 100

FFLAGS := -mp=gpu -gpu=$(SM)
ifeq ($(OPTIMIZE),yes)
FFLAGS += -O3
endif
ifeq ($(DEBUG),yes)
FFLAGS += -g
endif

program = main
source = main.f90

$(program): $(source)
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(program) *.o *.mod

run: $(program)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(program) $(RUN_ARGS)
