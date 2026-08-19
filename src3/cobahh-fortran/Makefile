FC = nvfortran
OPTIMIZE ?= yes
DEBUG ?= no
LAUNCHER ?=
RUN_ARGS ?= 1000000 10
PROGRAM := main
FFLAGS := -mp=gpu -gpu=cc90 -Mextend
ifeq ($(OPTIMIZE),yes)
FFLAGS += -O3
endif
ifeq ($(DEBUG),yes)
FFLAGS += -g
endif
$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
clean:
	rm -f $(PROGRAM) *.o *.mod
.PHONY: clean run
