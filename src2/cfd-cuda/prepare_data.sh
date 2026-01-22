#!/bin/sh

file1=../data/cfd/fvcorr.domn.097K.bz2
file_out=../data/cfd/fvcorr.domn.097K
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/cfd/fvcorr.domn.193K.bz2
file_out=../data/cfd/fvcorr.domn.193K
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/cfd/missile.domn.0.2M.bz2
file_out=../data/cfd/missile.domn.0.2M
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

