import os
from pathlib import Path
import pandas as pd



for path in Path('./zju-gaitacc').rglob('[1-5].csv'):
    pd.read_csv(path, header=None).T.to_csv(path, header=False, index=False)
#path='./zju-gaitacc'
#for filename in glob.iglob(path+'/*.txt', recursive=True):
 # print("ciao")
  #pre, ext = os.path.splitext(filename)
  #os.rename(filename, pre + ".csv")