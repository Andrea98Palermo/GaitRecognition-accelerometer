import glob
import csv
import os
from pathlib import Path
from os.path import join


i = 0
for path in sorted(Path('./zju-gaitacc').rglob('[1-5].csv')):
    if "result" not in str(path):
        path_ = str(path).split(os.sep)
        path_.insert(1, 'result')
        p = "/".join(path_)
        print(path)
        print(p)
        with open(path, 'w') as write_obj, open(join(p), 'r') as read_obj:
            writer = csv.writer(write_obj)
            reader = csv.reader(read_obj)
            for row in reader:
                row.insert(0, i)
                writer.writerow(row)
            i+=1

    