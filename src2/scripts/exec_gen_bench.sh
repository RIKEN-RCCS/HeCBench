#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./scripts/exec_gen_bench.sh BENCHMARK_LIST_FILE
  exit
fi
while read line
do echo ${line}
  ./scripts/gen_bench.sh ${line} ${2}
done < "${1}"

