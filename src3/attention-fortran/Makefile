FC = nvfortran
OPTIMIZE ?= yes
SM ?= cc90
LAUNCHER ?=
RUN_ARGS ?= 65536 2048 1000
RUN_ARGS_2 ?= 8192 8192 1000

FFLAGS = -O3 -mp=gpu -gpu=$(SM) -Mextend
ifeq ($(OPTIMIZE),no)
FFLAGS := -mp=gpu -gpu=$(SM) -Mextend
endif

main: main.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f main *.mod

run: main
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS_2)
