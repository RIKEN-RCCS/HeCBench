#!/bin/sh
file1=../data/leukocyte/testfile.avi.bz2_part_aa
file2=../data/leukocyte/testfile.avi.bz2_part_ab
#
fileM=../data/leukocyte/testfile.avi.bz2
#
file_out=../data/leukocyte/testfile.avi
#
if [ ! -e $file_out ]; then
    cat $file1 $file2 > $fileM
    bunzip2 $fileM
fi
