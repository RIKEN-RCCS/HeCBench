FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
LAUNCHER ?=
RUN_ARGS ?= ../data/bfs/graph1MW_6.txt
GRAPH := ../data/bfs/graph1MW_6.txt
GRAPH_ARCHIVE := $(GRAPH).bz2
main: main.f90
	$(FC) $(FFLAGS) -o $@ $<
$(GRAPH): $(GRAPH_ARCHIVE)
	bzip2 -dkf $<

run: main $(GRAPH)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main *.o *.mod
