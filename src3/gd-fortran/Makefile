FC ?= nvfortran
FFLAGS ?= -O3 -Minfo -mp=gpu -gpu=cc90 -Mextend
PROGRAM = main
DATA = gisette_scale

$(PROGRAM): gd_data.f90 main.f90
	$(FC) $(FFLAGS) -o $@ gd_data.f90 main.f90

$(DATA): ../gd-cuda/gisette_scale.bz2
	bzip2 -dkf $<
	mv ../gd-cuda/gisette_scale $@

run: $(PROGRAM) $(DATA)
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) $(DATA) 0.0001 10 100

clean:
	rm -f $(PROGRAM) *.mod $(DATA)
