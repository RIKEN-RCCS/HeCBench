FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM := main
SOURCES := bmp_mod.f90 sad_mod.f90 main.f90
LAUNCHER ?=

.PHONY: all run clean

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

../sad-cuda/coins.bmp ../sad-cuda/coin.bmp: ../sad-cuda/data.tar.gz
	tar -xzf $< -C ../sad-cuda/

run: $(PROGRAM) ../sad-cuda/coins.bmp ../sad-cuda/coin.bmp
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) ../sad-cuda/coins.bmp ../sad-cuda/coin.bmp 100
clean:
	rm -f $(PROGRAM) *.o *.mod
