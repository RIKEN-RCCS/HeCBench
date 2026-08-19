FC ?= nvfortran
FFLAGS ?= -O3 -Minfo -mp=gpu -gpu=cc90 -Mextend
PROGRAM = main
DATA = Face.pgm info.txt class.txt
DATA_STAMP = .face-data-ready
SOURCES = face_types.f90 face_image.f90 face_haar.f90 main.f90

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

$(DATA_STAMP): ../face-cuda/dataset.tar.gz
	tar -xzf $< -C .
	touch $@

$(DATA): $(DATA_STAMP)

run: $(PROGRAM) $(DATA)
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) Face.pgm info.txt class.txt Output-gpu.pgm

clean:
	rm -f $(PROGRAM) *.mod Output-gpu.pgm $(DATA) $(DATA_STAMP)
