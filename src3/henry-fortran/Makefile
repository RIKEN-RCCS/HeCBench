FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
DATA_DIR ?= ../henry-cuda
INPUT ?= $(DATA_DIR)/IRMOF-1.cssr
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
$(INPUT): $(DATA_DIR)/data.tar.gz
	tar -xzf $< -C $(DATA_DIR)
run: main $(INPUT)
	OMP_TARGET_OFFLOAD=MANDATORY ./main $(INPUT) 1000
clean:
	rm -f main *.mod
