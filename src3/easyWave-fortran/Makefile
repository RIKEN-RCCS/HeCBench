FC ?= nvfortran
FFLAGS ?= -O3 -Minfo -mp=gpu -gpu=cc90 -Mextend
PROGRAM = main
SOURCES = easywave_constants.f90 easywave_okada.f90 easywave_fault.f90 easywave_solver.f90 main.f90
GRID = ../easyWave-omp/data/grids/e2Asean.grd
FAULT = ../easyWave-omp/data/faults/BengkuluSept2007.flt

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) -grid $(GRID) -source $(FAULT) -time 120

check:
	gfortran -fopenmp -fsyntax-only $(SOURCES)

clean:
	rm -f $(PROGRAM) *.mod eWave.2D.sshmax eWave.2D.time
