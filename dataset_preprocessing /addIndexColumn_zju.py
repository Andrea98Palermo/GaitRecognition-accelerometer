import csv
import os.path
import os
from pathlib import Path
import sys
from os.path import join





for path in Path('zju-gaitacc').rglob('[1-5].csv'):
    if "result" not in str(path):
        path_ = str(path).split(os.sep)
        path_.insert(1, 'result')
        p = "/".join(path_)
        print(path)
        print(p)
        with open(path, 'r') as read_obj, open(join(p), 'w') as write_obj:
            
            writer = csv.writer(write_obj)
            reader = csv.reader(read_obj)
            index = 0
            for row in reader:
                row.insert(0, index)
                writer.writerow(row)
                index += 1

