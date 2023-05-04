import glob
import os
from pathlib import Path

for path in Path('./zju-gaitacc').rglob('*.txt'):
    pre, ext = os.path.splitext(path)
    os.rename(path, pre + ".csv")

#path='./zju-gaitacc'
#for filename in glob.iglob(path+'/*.txt', recursive=True):
 # print("ciao")
  #pre, ext = os.path.splitext(filename)
  #os.rename(filename, pre + ".csv")