#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./scripts/run_bench.sh BENCHMARK_NAME GPU_TYPE [run/run_mem] [timeout]
  exit
else
  echo "running ${1} on ${2}"
  echo "command "
  cat ./include/${1}-${2}
  run="run"
  if [ -n "${3}" ];then
    run="${3}"
  fi
  timeout_cmd=""
  if [ -n "${4}" ];then
    timeout_cmd=timeout
  fi
  if [[ "${2}" == "NVIDIA" ]];then
    if [ -e ${1}-cuda ];then
      echo "running benchmark under ${1}-cuda"
      cd ${1}-cuda
      ${timeout_cmd} ${4} make -f Makefile.NVD ${run} 1> log_run_bench.std 2> log_run_bench.err
      cd ..
    fi
    if [ -e ${1}-omp_nvc ];then
      echo "running benchmark under ${1}-omp_nvc"
      cd ${1}-omp_nvc
      ${timeout_cmd} ${4} make -f Makefile.NVD ${run} 1> log_run_bench.std 2> log_run_bench.err
      cd ..
    fi
  elif [[ "${2}" == "AMD" ]];then
    echo "running benchmark under ${1}-hip"
    if [ -e ${1}-hip ];then
      cd ${1}-hip
      ${timeout_cmd} ${4} make -f Makefile.AMD ${run} 1> log_run_bench.std 2> log_run_bench.err
      cd ..
    fi
    if [ -e ${1}-hipified ];then
      echo "running benchmark under ${1}-hipified"
      cd ${1}-hipified
      ${timeout_cmd} ${4} make -f Makefile.AMD ${run} 1> log_run_bench.std 2> log_run_bench.err
      cd ..
    fi
    if [ -e ${1}-omp_aomp ];then
      echo "running benchmark under ${1}-omp_aomp"
      cd ${1}-omp_aomp
      ${timeout_cmd} ${4} make -f Makefile.AMD ${run} 1> log_run_bench.std 2> log_run_bench.err
      cd ..
    fi
  else
    echo "the 2nd argument must be either NVIDIA or AMD"
    exit
  fi
fi
