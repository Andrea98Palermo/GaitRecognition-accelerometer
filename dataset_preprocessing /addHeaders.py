import pandas as pd
import glob
import csv

path='./flat'
for filename in glob.iglob(path+'/*.csv'):
    
    with open(filename,newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open(filename,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','time', 'a_x', 'a_y', 'a_z'])
        w.writerows(data)

