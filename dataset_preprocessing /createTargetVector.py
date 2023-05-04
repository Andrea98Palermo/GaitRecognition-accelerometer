import glob
import csv
import os

path = './flat'
allFiles = glob.glob(path + "/*.csv")
allFiles.sort()
with open('targetVector.csv', 'w') as outfile:
    w = csv.writer(outfile)
    for fname in allFiles:
      #print(str(os.path.basename(fname)[2:5]))
      w.writerow([str(os.path.basename(fname)[2:5])])