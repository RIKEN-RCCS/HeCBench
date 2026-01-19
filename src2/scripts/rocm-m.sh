#!/bin/sh
maxm=$( cat log.mem | awk '{if(m<$8) m=$8} END{print m}' )
maxm=$( printf %.2f $( echo ${maxm}/1000000 | bc -l ) )
minm=$( cat log.mem | awk 'BEGIN{m="inf"}{if(m>$8) m=$8} END{print m}' )
minm=$( printf %.2f $( echo ${minm}/1000000 | bc -l ) )
diffm=$( printf %.2f $( echo ${maxm}-${minm} | bc -l ) )
echo "min max diff memory in [MB] : ${minm} ${maxm} ${diffm}"
