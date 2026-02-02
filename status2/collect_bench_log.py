#!/usr/bin/env python3

import regexps
import optparse
import os
import sys
import re

archs = ['cuda', 'hip', 'hipified', 'omp_nvc', 'omp_aomp']

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
    print(str(res))
    return res

form='{:.2e}'
def run():
    parser = optparse.OptionParser(usage="%prog [options]",description="collect data from benchmark log files")
    parser.add_option('-b', '--bench_names', type=str, dest='bench_names', default='bench_names')
    parser.add_option('-o', '--output', type=str, dest='output', default='log_std.md')
    (options, args) = parser.parse_args()
    fw = open(options.output,'w')
    fw.write('| name | cuda | hip | hipified | omp_nvc | omp_aomp |\n')
    fw.write('|  --  |  --  | --  |   --     |   --    |    --    |\n')
    f=open(options.bench_names)
    for line in f:
        line = line.strip()
        if line.startswith('#') or len(line)==0:
            continue
        rexp = regexps.regexps[line]
        if rexp is None or len(rexp.strip())==0:
            print('Regular expression for '+line+' is undefined ')
        row = '|'+line+'|'
        for arch in archs:
            bDir = line+'-'+arch
            if os.path.exists(bDir+'/log.std'):
                res = parse_log_std(bDir+'/log.std', rexp)
                if res is None:
                    row += ' |'
                    print('invalid Benchmark logfile under '+bDir)
                else:
                    row += form.format(res)+' |'
            else:
                print('log.std file does not exist under '+line+'-'+arch)
                row += ' |'
        fw.write(row+'\n')
    f.close()
    fw.close()

if __name__ == '__main__':
    run()
