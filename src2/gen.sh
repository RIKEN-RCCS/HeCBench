#!/bin/sh
for n in `cat List_full`
do
    ./scripts/gen_bench.sh $n
done
