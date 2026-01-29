#!/bin/sh
OMP_DIR=$1
ACC_DIR=$2
#
rm -rf $ACC_DIR
mkdir $ACC_DIR
#
for file in "$OMP_DIR"/*; do
    filename=$(basename "$file")
    target="$ACC_DIR/$filename"
    
    if [ -f "$file" ]; then
        # ソースファイルの場合は変換
        if [[ "$filename" == *.cpp ]] || [[ "$filename" == *.c ]] || [[ "$filename" == *.h ]] || [[ "$filename" == *Makefile* ]]; then
            echo "    変換: $filename"
            sed -e 's/#pragma omp target data map(/#pragma acc data /' \
                -e 's/#pragma omp target update to/#pragma acc update to/' \
                -e 's/#pragma omp target update from/#pragma acc update from/' \
                -e 's/#pragma omp target teams distribute parallel for/#pragma acc parallel loop/' \
                -e 's/#pragma omp target teams distribute/#pragma acc parallel loop/' \
                -e 's/#pragma omp target parallel for/#pragma acc parallel loop/' \
                -e 's/#pragma omp target/#pragma acc parallel/' \
                -e 's/#pragma omp parallel for reduction(/#pragma acc loop reduction(/' \
                -e 's/#pragma omp parallel for/#pragma acc loop/' \
                -e 's/#pragma omp atomic update/#pragma acc atomic update/' \
                -e 's/#pragma omp atomic/#pragma acc atomic/' \
                -e 's/num_teams(\([0-9]*\))/gang(\1)/' \
                -e 's/num_threads(\([0-9]*\))/vector(\1)/' \
                -e 's/thread_limit(\([0-9]*\))/vector_length(\1)/' \
                -e 's/#include <omp\.h>/#include <openacc.h>/' \
                -e 's/-fiopenmp/-fopenacc/' \
                -e 's/-qopenmp/-acc/' \
                -e 's/-fopenmp/-fopenacc/' \
                "$file" > "$target"
        else
            echo "    コピー: $filename"
            cp "$file" "$target"
        fi
    fi
done
