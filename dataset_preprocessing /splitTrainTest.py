from pathlib import Path
from shutil import copyfile
import os


for path in Path('./result').rglob('*[1-2].csv'):
    p = str(path)
    print(p)
    if p.split(os.sep)[1].startswith("1"):
        copyfile(p, os.path.dirname(p) + "/../training/" + path.name)


for path in Path('./result').rglob('*[3-4].csv'):
    p = str(path)
    print(p)
    if p.split(os.sep)[1].startswith("2"):
        copyfile(p, os.path.dirname(p) + "/../training/" + path.name)        

for path in Path('./result').rglob('*[3-6].csv'):
    p = str(path)
    print(p)
    if p.split(os.sep)[1].startswith("1"):
        copyfile(p, os.path.dirname(p) + "/../testing/" + path.name)

for path in Path('./result').rglob('*[1-2].csv'):
    p = str(path)
    print(p)
    if p.split(os.sep)[1].startswith("2"):
        copyfile(p, os.path.dirname(p) + "/../testing/" + path.name)

for path in Path('./result').rglob('*[5-6].csv'):
    p = str(path)
    print(p)
    if p.split(os.sep)[1].startswith("2"):
        copyfile(p, os.path.dirname(p) + "/../testing/" + path.name)