#!/bin/sh
nl=$( wc log.mem| awk '{print $1}' )
nl1=$( echo "$nl-1" | bc -l )
maxm=$( tail -${nl1} log.mem | awk '{if(m<$1) m=$1} END{print m}' )
minm=$( tail -${nl1} log.mem | awk 'BEGIN{m="inf"}{if(m>$1) m=$1} END{print m}' )
diffm=$( echo ${maxm}-${minm} | bc -l )
echo "min max diff memory in [MB] : ${minm} ${maxm} ${diffm}"
