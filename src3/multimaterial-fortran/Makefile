FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= 2000 2000
SOURCES = multimat_types.f90 full_matrix.f90 compact.f90 multimat.f90
OBJECTS = $(SOURCES:.f90=.o)

all: multimat multimat_F multimat_FL

multimat: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

multimat_F: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

multimat_FL: $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(OBJECTS)

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

run: all
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat_FL 2000 2000
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat_F 2000 2000
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat 2000 2000
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat_FL 2000 2000 0.3 0.05 0.05
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat_F 2000 2000 0.3 0.05 0.05
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./multimat 2000 2000 0.3 0.05 0.05
clean:
	rm -f multimat multimat_F multimat_FL *.o *.mod
