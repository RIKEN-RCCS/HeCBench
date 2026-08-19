FC ?= nvfortran
OPTIMIZE ?= yes
DEBUG ?= no
SM ?= cc90
LAUNCHER ?=
RUN_ARGS ?=

PROGRAM := main
SOURCE := main.f90
FFLAGS := $(EXTRA_FFLAGS) -Mextend -mp=gpu -gpu=$(SM)

ifeq ($(OPTIMIZE),yes)
  FFLAGS += -O3
endif

ifeq ($(DEBUG),yes)
  FFLAGS += -g
endif

$(PROGRAM): $(SOURCE)
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(PROGRAM) *.o *.mod

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
