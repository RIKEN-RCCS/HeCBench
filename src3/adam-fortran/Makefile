FC = nvfortran
FFLAGS = -O3 -mp=gpu -gpu=cc90 -Mextend
TARGET = main
LAUNCHER ?=
RUN_ARGS ?= 10000 200 100
all: $(TARGET)
$(TARGET): main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) $(RUN_ARGS)
clean:
	rm -f $(TARGET) *.o *.mod
