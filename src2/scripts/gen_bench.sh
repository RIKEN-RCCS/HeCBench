#!/bin/bash

if [ -z "${1}" ];then
  echo Usage : ./gen_bench.sh BENCHMARK_NAME
  exit
else
  if [ -e ${1}-sycl ];then
    cd ${1}-sycl
    pwd
    ../scripts/edit_makefile.py -t NVIDIA --command
    cat Makefile.NVD | sed -e 's/sm_60/sm_90/g' | sed -e 's/sm_70/sm_90/g' > TMPFILE
    mv TMPFILE Makefile.NVD
    cd ..
  fi
  if [ -e ${1}-omp ];then
      ./scripts/omp2acc.sh ${1}-omp ${1}-acc

      if [ -e ${1}-omp/Makefile.nvc ];then
	  cd ${1}-acc
	  pwd
	  ../scripts/edit_makefile.py -t NVIDIA -i Makefile.nvc
	  cat Makefile.NVD | sed -e 's/cc70/cc90/g' | sed -e 's/nvc++/nvc++ -acc/g' > TMPFILE
	  mv TMPFILE Makefile.NVD
	  cd ..

	  cp -R ${1}-omp ${1}-omp_nvc
	  cd ${1}-omp_nvc
	  pwd
	  ../scripts/edit_makefile.py -t NVIDIA -i Makefile.nvc
	  cat Makefile.NVD | sed -e 's/cc70/cc90/g' > TMPFILE
	  mv TMPFILE Makefile.NVD
	  cd ..
      fi
  fi
fi
