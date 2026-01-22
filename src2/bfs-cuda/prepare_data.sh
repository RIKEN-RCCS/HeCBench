#!/bin/sh

file1=../data/bfs/graph1MW_6.txt.bz2
file_out=../data/bfs/graph1MW_6.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/bfs/graph4096.txt.bz2
file_out=../data/bfs/graph4096.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

file1=../data/bfs/graph65536.txt.bz2
file_out=../data/bfs/graph65536.txt
#
if [ ! -e $file_out ]; then
    bunzip2 $file1
fi

