import glob
import csv
import os.path

path='./OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter'
for filename in glob.iglob(path+'/*.csv'):
  with open(filename, 'wb') as fp_out, open('./OU-IneritialGaitData/result/'+os.path.basename(filename), "r") as fp_in:
    reader = csv.reader(fp_in, delimiter=",")
    writer = csv.writer(fp_out, delimiter=",")
    for row in reader:
        if len(row) >= 4:
          del row[3:]
        writer.writerow(row)