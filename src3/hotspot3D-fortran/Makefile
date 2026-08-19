FC = nvfortran
FFLAGS ?= -O3 -Minfo -mp=gpu -gpu=cc90 -Mextend
PROGRAM = main
DATA_DIR = ../data/hotspot3D

all: $(PROGRAM)

$(PROGRAM): main.f90
	$(FC) $(FFLAGS) -o $@ $<

run: $(PROGRAM) $(DATA_DIR)/power_512x8 $(DATA_DIR)/temp_512x8
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) 512 8 5000 $(DATA_DIR)/power_512x8 $(DATA_DIR)/temp_512x8 output.out

$(DATA_DIR)/power_512x8: $(DATA_DIR)/power_512x8.bz2
	bzip2 -dkf $<

$(DATA_DIR)/temp_512x8: $(DATA_DIR)/temp_512x8.bz2
	bzip2 -dkf $<

test: $(PROGRAM) $(DATA_DIR)/power_64x8 $(DATA_DIR)/temp_64x8
	OMP_TARGET_OFFLOAD=MANDATORY ./$(PROGRAM) 64 8 2 $(DATA_DIR)/power_64x8 $(DATA_DIR)/temp_64x8 test.out

$(DATA_DIR)/power_64x8: $(DATA_DIR)/power_64x8.bz2
	bzip2 -dkf $<

$(DATA_DIR)/temp_64x8: $(DATA_DIR)/temp_64x8.bz2
	bzip2 -dkf $<

clean:
	rm -f $(PROGRAM) *.mod output.out test.out
