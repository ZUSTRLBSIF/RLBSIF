#This script is used to correct the format errors caused during the protonation process.

import os
#pmodify_pdbqt.pydbqtdir = ''
#newpdbqtdir = ''
list = os.listdir(pdbqtdir)
for pdbqt in list:
    print(pdbqt)
    pdb_filename = pdbqtdir + pdbqt
    file = open(pdb_filename, "r")
    lines = file.readlines()
    newfile = open(newpdbqtdir + pdbqt, "w")
    for i in range(len(lines)):
        line = lines[i]
        #if line[12] == "'" and line[14] == 'O':
        if line[12] == "'":
            #print(line)
            line = line.replace(line[12]," ")
            #line = line.replace(line[15],"'")
            #print(line)
            newfile.write(line)
        else:
            newfile.write(line)





