FC ?= nvfortran
OPTFLAGS ?= -O3
OMPFLAGS ?= -mp=gpu -gpu=cc90 -Mextend
TARGET ?= main
SOURCE := main.f90
LAUNCHER ?=
RUN_ARGS ?= 8200 8100 100

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(FC) $(OPTFLAGS) $(OMPFLAGS) -o $@ $<

run: $(TARGET)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(TARGET) $(RUN_ARGS)

clean:
	rm -f $(TARGET) *.o *.mod
