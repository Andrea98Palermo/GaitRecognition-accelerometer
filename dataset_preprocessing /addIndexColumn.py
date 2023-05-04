import glob
import csv
import os.path

path='./flat'
for filename in glob.iglob(path+'/*.csv'):
    with open(filename, 'r') as read_obj, open('./result/'+os.path.basename(filename), 'w') as write_obj:
        print(filename)
        writer = csv.writer(write_obj)
        reader = csv.reader(read_obj)
        index = 0
        for row in reader:
            row.insert(0, index)
            writer.writerow(row)
            index += 1