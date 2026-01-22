import glob
benches = sorted(list(map(lambda x:x.split('-cuda')[0], glob.glob('*-cuda'))))
for b in benches:
    print(b)
