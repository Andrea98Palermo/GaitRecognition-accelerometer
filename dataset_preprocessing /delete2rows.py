import glob

path='./OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter'
for filename in glob.iglob(path+'/*.csv'):
  with open(filename, 'r') as f:
    lines = f.read().split("\n")
    f.close()
    if len(lines) >= 3:
      lines = lines[:-2]
      o = open(filename, 'w')
      for line in lines:
        o.write(line+'\n')
      o.close()