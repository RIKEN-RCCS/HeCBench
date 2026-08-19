FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= ../sw4ck-cuda/sw4ck.in 100
PROGRAM := main
SOURCES := utils.f90 sw4_stencil.f90 kernel1.f90 kernel2.f90 kernel3.f90 kernel4.f90 kernel5.f90 curvilinear4sg.f90 main.f90

.PHONY: all clean run
all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) ../sw4ck-cuda/sw4ck.in 100
clean:
	rm -f $(PROGRAM) *.o *.mod
