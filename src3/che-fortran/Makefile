FC       ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend -Minfo
LAUNCHER ?=
RUN_ARGS ?= 1000

.PHONY: all clean run
all: main

main: main.f90
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $<

run: main
	rm -rf out
	mkdir -p out
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -rf main *.o *.mod out
