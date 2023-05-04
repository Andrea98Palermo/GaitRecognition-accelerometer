import os.path
from pathlib import Path
import csv



for path in Path('./zju-gaitacc').rglob('3.csv'):
    p = str(path)
    if "session_0" not in p:
        out_path = os.path.dirname(p) + "/../../../../result/" + p[20] + "_" + p[27] + p[28] + p[29] + "_" + p[35] + ".csv"
        with open(os.path.dirname(p) + "/useful.csv", 'r') as useful_f, open(path, 'r') as src, open(out_path, 'w') as out:
            useful_reader = csv.reader(useful_f)
            borders = next(useful_reader)
            print(borders)
            f_reader =  csv.reader(src)
            i = 0
            writer = csv.writer(out)
            for row in  f_reader:
                if i >= int(borders[0]) or i <= int(borders[1]):
                    writer.writerow(row)
                i+=1
                    


        

