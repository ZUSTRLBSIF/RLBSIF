#extract ID
import os
dir = ''
list = os.listdir(dir)
files = open('', "w")
for pdb in list:
    pdb_filename = dir + pdb
    file = open(pdb_filename, "r")
    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if line[0:6] == 'HETATM':
            print(pdb)
            print(line)
            print(line[17:20])
            files.write(line[17:20] + " " + pdb + '\n')
            break




