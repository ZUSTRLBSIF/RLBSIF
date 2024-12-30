#format the file


import os


def pdbqt_to_pdb(in_pdbqt_file, out_pdb_file):
    cmd = "obabel -ipdbqt " + in_pdbqt_file + " -O " + out_pdb_file
    print(cmd)
    os.system(cmd)