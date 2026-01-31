#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./scripts/run_bench.sh BENCHMARK_NAME [run/run_mem] [timeout]
  exit
else
  echo "running ${1}"
  echo "command "
  cat ./include/${1}-NVIDIA
  run="run"
  if [ -n "${2}" ];then
      run="${2}"
  fi
  timeout_cmd=""
  if [ -n "${3}" ];then
      timeout_cmd=timeout
  fi

  if [ -e ${1}-sycl ];then
      echo "running benchmark under ${1}-sycl"
      cd ${1}-sycl
      ${timeout_cmd} ${3} make -f Makefile.NVD ${run} 1> log_run_bench.std 2> log_run_bench.err
      if [ "${timeout_cmd}" != "" ]; then
	  echo "timeout was set to " ${3} >> log_run_bench.err
      fi
      cd ..
  fi
  if [ -e ${1}-acc ];then
      echo "running benchmark under ${1}-acc"
      cd ${1}-acc
      ${timeout_cmd} ${3} make -f Makefile.NVD ${run} 1> log_run_bench.std 2> log_run_bench.err
      if [ "${timeout_cmd}" != "" ]; then
	  echo "timeout was set to " ${3} >> log_run_bench.err
      fi
      cd ..
  fi
  if [ -e ${1}-omp_nvc ];then
      echo "running benchmark under ${1}-omp_nvc"
      cd ${1}-omp_nvc
      ${timeout_cmd} ${3} make -f Makefile.NVD ${run} 1> log_run_bench.std 2> log_run_bench.err
      if [ "${timeout_cmd}" != "" ]; then
	  echo "timeout was set to " ${3} >> log_run_bench.err
      fi
      cd ..
  fi
fi
