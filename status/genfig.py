#!/usr/bin/env python3

import optparse
import os
import sys
from matplotlib import pyplot as plt

def include_str(incfile):
    f = open(incfile)
    ret = ''
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        ret += line
    f.close()
    #ar = ret.split('=')
    ar = ret.split('COMMAND =')
    if len(ar)>1:
        ret = ar[1].strip()
    ret = ret.split(' )')[0]
    ar = ret.split('( ')
    if len(ar)>1:
        ret = ar[1]
    else:
        return ret
    
    ret = ret.split('1>')[0].strip()

    ar = ret.split()
    if len(ar)>1:
        ret = ' '.join(ar[1:len(ar)])
    else:
        ret = ''

    return ret.strip()

def compare_command(srcdir, bench_name):
    nvinc  = srcdir+'/include/'+bench_name+'-NVIDIA'
    amdinc = srcdir+'/include/'+bench_name+'-AMD'
    if not os.path.exists(nvinc) or not os.path.exists(amdinc):
        print('include file does not exist for '+bench_name)
        return ''
    inv = include_str(nvinc)
    iam = include_str(amdinc)
    if inv != iam:
        #return '`'+inv+'`\n | | | | | | | | `AMD    '+iam+'`'
        return '`'+inv + '`\n | | | | | | | | '+'`'+iam+'`'
    else:
        return ''

def extend_markdown(markd, imagedir='SVGs', src_dir='../src2', mark_command=False):
    f = open(markd)
    firstline = f.readline().strip()
    firstline += ' plot |'
    if mark_command:
        firstline += ' command |'
    print(firstline)
    secondline = f.readline().strip()
    secondline += ' -- |'
    if mark_command:
        secondline += ' -- |'
    print(secondline)
    for line in f:
        line = line.strip()
        bname = line.split('|')[1].strip()
        line += '!['+bname+']('+imagedir+'/'+bname+'.svg) |'
        if mark_command:
            line += compare_command(src_dir, bname)+'|'
        print(line)
    f.close()

def gen_plt(x, line, imagedir='SVGs'):
    words = line.strip().split('|')
    bname = words[1].strip()
    words = words[2:len(words)-1]
    y = []
    for w in words:
        w = w.strip()
        try:
            fl = float(w)
        except ValueError:
            fl = 0.0
        y.append(fl)
    plt.close()
    fig, ax = plt.subplots()
    ax.set_ylabel('time (s)')
    plt.bar(x, y)
    os.makedirs(imagedir, exist_ok = True)
    outf = imagedir+'/'+bname+'.svg'
    plt.savefig(outf)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="%prog [options]",description="generate bar graph from the markdown table")
    parser.add_option('-m', '--markdown', dest='markdown', default=None)
    parser.add_option('-e', '--extend', dest='extend', action='store_true', default=False)
    parser.add_option('-n', '--noplot', dest='noplot', action='store_true', default=False)
    parser.add_option('-i', '--imagedir', dest='imagedir', default='SVGs')
    parser.add_option('--mark_command', dest='mark_command', action='store_true', default=False)
    parser.add_option('--src_dir', dest='src_dir', default='../src2')

    (options, args) = parser.parse_args()

    if options.markdown is None or not os.path.exists(options.markdown):
        print('the specified markdown file does not exist')
        sys.exit()
    f = open(options.markdown)
    line0 = f.readline()
    columns = line0.strip().split('|')
    x = columns[2:len(columns)-1]
    dumm = f.readline()

    if not options.noplot:
        for line in f:
            line = line.strip()
            gen_plt(x,line,imagedir=options.imagedir)
    f.close()

    if options.extend:
        extend_markdown(options.markdown, imagedir=options.imagedir, src_dir=options.src_dir, mark_command=options.mark_command)

