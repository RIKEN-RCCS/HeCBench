FC = nvfortran
OPTIMIZE ?= yes
DEBUG ?= no
LAUNCHER ?=
RUN_ARGS ?=

PROGRAM := main
SOURCE := main.f90
FFLAGS := -mp=gpu -gpu=cc90 -Mextend

ifeq ($(OPTIMIZE),yes)
FFLAGS += -O3
endif
ifeq ($(DEBUG),yes)
FFLAGS += -g
endif

$(PROGRAM): $(SOURCE)
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod

.PHONY: clean run
