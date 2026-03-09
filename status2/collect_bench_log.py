#!/usr/bin/env python3

import regexps
import optparse
import os
import sys
import re

archs = ['cuda', 'sycl', 'acc', 'omp_nvc']

def parse_log_std(logstd, rexp):
    if rexp is None or len(rexp)==0:
        return None
    stdstr = open(logstd).readlines()
    out = ' '.join(stdstr)
    out = out.strip()
    res = None
    try:
        res = re.findall(rexp, out)
    except re.error as e:
        print("Regular expression error occurred:", e.msg)
        print("Pattern:", e.pattern)
        print("Position:", e.pos)
    if res is None or not res:
        print("no regex match for " + rexp + " in\n" + logstd)
        return None
    res = sum([float(i) for i in res]) #in case of multiple outputs sum them (e.g. total time)
#    print(str(res))
    return res

form='{:.2e}'
def run():
    parser = optparse.OptionParser(usage="%prog [options]",description="collect data from benchmark log files")
    parser.add_option('-b', '--bench_names', type=str, dest='bench_names', default='bench_names')
    parser.add_option('-o', '--output', type=str, dest='output', default='log_std.md')
    (options, args) = parser.parse_args()

    fw = open(options.output,'w')
    fw.write('# ベンチマークが標準出力に出力する値 \n')
    fw.write('\n')
    fw.write('| 名称 | cuda | sycl | acc | omp  | 単位 | 分類 | \n')
    fw.write('|  ---: | ---:  | ---: | ---: | ---: | ---:   | ---:   | \n')

    f=open(options.bench_names)
    
    for line in f:
        line = line.strip()

        f2 = open( "List_omp_get" )
        myclass = "A"
        for line2 in f2:
            if line == line2.strip():
                myclass = "B"; break
        f2.close()
        
        if line.startswith('#') or len(line)==0:
            continue
        rexp = regexps.regexps[line]

        tmp1 = rexp.split(")(");  ndata = len(tmp1)
        if ndata == 3:
#            print("tmp1  ",tmp1[2])
            unit = tmp1[2].replace("?: ","")
            unit = unit.replace("?:","")
            unit = unit.replace("\\","")
            unit = unit.replace("(","")
            unit = unit.replace(")","")
            unit = unit.replace("seconds", "s")
            unit = unit.replace("seconds.", "s")
            unit = unit.replace("secs", "s")
            unit = unit.replace("s]", "s")
            unit = unit.replace("s.", "s")
            unit = unit.replace("usec", "us")
            unit = unit.replace("msec", "ms")
            unit = unit.replace("sec", "s")
            unit = unit.replace("GBytes/sec", "GB/s")
            unit = unit.replace("GBytes/s", "GB/s")
            unit = unit.replace("GFLOPS/s", "GFLOPS")

            if line == "adv": unit = "ns"
            if line == "bwt": unit = "ms"
            if line == "cooling": unit = "ms"
            if line == "cross": unit = "us"
            if line == "damage": unit = "s"
            if line == "d2q9-bgk": unit = "us"
            if line == "fdtd3d": unit = "s"
            if line == "feynman-kac": unit = "s"
            if line == "phmm": unit = "ms"
            if line == "rsc": unit = "ms"
            if line == "sw4ck": unit = "ms"
            if line == "testSNAP": unit = "ms"
            if line == "tqs": unit = "ms"

        else:
            unit = ""
            if line == "assert" : unit = "ns"
            if line == "axhelm" : unit = "ns"
            if line == "babelstream" : unit = "s"
            if line == "ccsd-trpdrv" : unit = "s"
            if line == "crc64": unit = "MB/s"
            if line == "heat": unit = "s"
            if line == "heat2d": unit = "GFLOPS"
            if line == "mdh": unit = "s"
            if line == "sad": unit = "ms"
            if line == "simpleSpmv": unit = "ms"
            if line == "snake": unit = "us"
#            print("XXXXXXX",line, unit)
            
        if rexp is None or len(rexp.strip())==0:
            print('Regular expression for '+line+' is undefined ')
        row = '| '+line+' |'

        for arch in archs:
#            bDir = line+'-'+arch
            bDir = "../src2/" +line+'-'+arch

            if line == "fpdc":
                target = bDir +"/log.err"
            else:
                target = bDir +"/log.std"
                
            if os.path.exists( target ):
                res = parse_log_std( target, rexp )
                if res is None:
                    row += ' |'
                    print('invalid Benchmark logfile under '+bDir)
                else:
                    row += ' ' +form.format(res)+' |'
            else:
                print('log.std file does not exist under '+line+'-'+arch)
                row += ' |'

        row += ' ' +unit +' | ' +myclass +' |'
        fw.write(row+'\n')

    fw.write("\n")
    fw.write("※ 分類: OpenMP コードが (omp_get_wtime 以外の) omp_get_* を (A) 含まない (B) 含む\n")
    fw.write("\n")
    fw.write("※ メモリ: cuda 版での値\n")
    fw.close()
    fw.close()

if __name__ == '__main__':
    run()
