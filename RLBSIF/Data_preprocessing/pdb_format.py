#This script is used to format PDB files.
import os
inputdir = ''
outputdir = ''
pdbname_list = os.listdir(inputdir)




for pdb_id in pdbname_list:
    pdb_filename = inputdir + pdb_id
    new_pdb_filename = outputdir + pdb_id
    file = open(pdb_filename, "r")
    newfile = open(new_pdb_filename, "w")
    lines = file.readlines()


    for i in range(len(lines)):
        line = lines[i]
        if (line[0:6] == 'CRYST1') or (line[0:4] == 'ATOM') or (line[0:3] == 'END'):
            print("ok")
        else:
            continue
        if (len(line) > 72) and (line[21] == 'A' or line[21] == 'B' or line[21] == 'C' or line[21] == 'X' or line[21] == 'Z' or line[21] == 'E' or line[21] == 'Y' or line[21] == 'D'):
            str = list(line)
            str[21] = 'R'
            line = "".join(str)
            newfile.write(line)
        else:
            newfile.write(line)








