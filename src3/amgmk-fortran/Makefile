FC       = nvfortran
SM       ?= cc90
PROGRAM  := AMGMk
SOURCES  := amgmk_types.f90 vector.f90 csr_matrix.f90 laplace.f90 csr_matvec.f90 relax.f90 main.f90
FFLAGS   ?= -O3 -Minfo -mp=gpu -gpu=$(SM) -Mextend
LAUNCHER ?=
RUN_ARGS ?=

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.mod *.o
