#!/bin/sh
file1=../hogbom-cuda/data/dirty_4096.img.bz2
file_out=../hogbom-cuda/data/dirty_4096.img
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../hogbom-cuda/data/psf_4096.img.bz2
file_out=../hogbom-cuda/data/psf_4096.img
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

