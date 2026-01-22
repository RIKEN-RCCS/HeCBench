#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./scripts/exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE [run/run_mem] [timeout]
  exit
fi

GPU=${2}
if [ -z "${GPU}" ];then
  echo Usage : ./scripts/exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE [run/run_mem] [timeout]
  exit
fi
run_mem=${3}
to=${4}
while read line
do echo ${line}
  ./scripts/run_bench.sh ${line} ${GPU} ${run_mem} ${to}
done < "${1}"

