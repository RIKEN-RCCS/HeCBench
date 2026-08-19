FC ?= nvfortran
FFLAGS ?= -O3 -mp=gpu -gpu=cc90 -Mextend
DATA=../data/kmeans/kdd_cup
$(DATA): ../data/kmeans/kdd_cup.tar.bz
	tar -xjf $< -C ../data/kmeans
kmeans: main.f90
	$(FC) $(FFLAGS) -o $@ $<
run: kmeans $(DATA)
	OMP_TARGET_OFFLOAD=MANDATORY ./kmeans -r -n 5 -m 15 -l 10 -o -i $(DATA)
clean:
	rm -f kmeans *.o *.mod
