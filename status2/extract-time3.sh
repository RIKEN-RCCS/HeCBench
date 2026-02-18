#!/bin/sh
echo "# 実行時間 (sec.)"
echo ""
echo "|" name "|" cuda "|" sycl "|" acc  "|"  omp_nvc "|"  分類 "|"
echo "|" "--" "|" "--" "|" "--" "|" "--" "|"   "--"   "|"  "--" "|"

##############################
SRCDIR=../src2

######################
sum0=0
sum1=0
sum2=0
sum3=0

sum_both=0

##############################
for n in `cat List_full`
do

    has_omp_func=0
    for m in `cat List_omp_get`
    do
	if [ "$n" = "$m" ]; then
	    has_omp_func=1
	    break
	fi
    done

    num=0
    while [ $num -lt 4 ];
    do
	if [ $num = 0 ]; then dir=$SRCDIR/$n-cuda; fi
	if [ $num = 1 ]; then dir=$SRCDIR/$n-sycl; fi
	if [ $num = 2 ]; then dir=$SRCDIR/$n-acc; fi
	if [ $num = 3 ]; then dir=$SRCDIR/$n-omp_nvc; fi
	
	if [ $num = 0 ]; then comment="| "$n" |"; fi
	if [ $num = 0 ]; then flag1=0; fi

	       
	if [ -d $dir ]; then
	    c1=`grep Error $dir/log_run_bench.err | grep make`
	    c1a=`echo $c1 | awk '{print $NF}'`
	    if [ -e $dir/log.build ]; then
		c1=`grep Error $dir/log.build | grep make`
	    fi

	    if [ "$c1"  = "" ]; then
		c2=`grep -i error $dir/log.err`
                c3=`grep "core dumped" $dir/log.time`

		c4=`grep Terminated $dir/log_run_bench.err`
		c5=`grep "timeout was set" $dir/log_run_bench.err`
		c6=`grep Error $dir/log_run_bench.err | grep make`

		if [ "$c4" != "" ]; then
		    if [ "$c5" != "" ]; then
			tmp1=`echo $c5 | awk '{print $NF}'`
			time="over "$tmp1
		    else
			time="TLE error"
		    fi
		elif [ "$c6" != "" ]; then
		    time="exe error"
		else
                    if [ "$c2"  = "" -a "$c3" = "" ]; then
			time=`grep real $dir/log.time | awk '{print $NF}'`
			if [ "$time" != "" ]; then
			    if [ $num = 0 ]; then let sum0="$sum0 +1"; fi
			    if [ $num = 1 ]; then let sum1="$sum1 +1"; fi
			    if [ $num = 2 ]; then let sum2="$sum2 +1"; fi
			    if [ $num = 3 ]; then let sum3="$sum3 +1"; fi

			    if [ $num = 1 ]; then flag1=1; fi
			    if [ $flag1 = 1 ]; then
				if [ $num = 2 ]; then let sum_both="$sum_both +1"; fi
			    fi
			fi
		    else
			time="exe err"
		    fi
		fi
	    else
		if [ $c1a -gt 100 ]; then
		    time="exe err"
		else
		    time="build err"
		fi
	    fi
	else
	    time="--"
	fi
	#
	comment=$comment" "$time" |"
	let num="$num +1"
    done

    if [ "$has_omp_func" = "0" ]; then
	class="A"
    else
	class="B"
    fi
    echo $comment $class "|"
done
######################
echo "|"           "|"        "|"       "|"       "|"      "|"    "|"
echo "|" "completed" "|"  $sum0 "|" $sum1 "|" $sum2 "|"  $sum3  "|"  "|"
echo ""
echo "sycl と acc がともに完了した件数" $sum_both
