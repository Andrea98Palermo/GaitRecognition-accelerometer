import glob
import csv
import os.path

path='./flat'
i = 0
for filename in sorted(glob.iglob(path+'/*.csv')):
    print(filename)
    with open(filename, 'w') as write_obj, open('./result/'+os.path.basename(filename), 'r') as read_obj:
        writer = csv.writer(write_obj)
        reader = csv.reader(read_obj)
        for row in reader:
            row.insert(0, i)
            writer.writerow(row)
        i+=1