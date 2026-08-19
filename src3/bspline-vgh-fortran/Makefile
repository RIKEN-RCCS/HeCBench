FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
TARGET = main
SOURCE = main.f90
LAUNCHER ?=
RUN_ARGS ?=

$(TARGET): $(SOURCE)
	$(FC) $(FFLAGS) -o $@ $<

run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) $(RUN_ARGS)

clean:
	rm -f $(TARGET) *.o *.mod
