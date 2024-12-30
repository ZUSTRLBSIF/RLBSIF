import os


pdbdir = ''
pocketdir = ''
list = os.listdir(pdbdir)


for pdb_id in list:
    pdb_filename = pdbdir + pdb_id
    out_name = pocketdir +pdb_id
    file = open(pdb_filename, "r")
    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if (line[0:6]) in ("HETATM"):
            name = line[17:20]
            str(name)
            name.strip()
            break






    cmd.load(pdb_filename)
    cmd.select(pdb_id,'resn'+' '+name)
    a = cmd.centerofmass(pdb_id)
    print(a)
    cmd.select()
    #cmd.(create pocket,pdb_id within 5 of resn a)
    cmd.save(out_name,pdb_id[0:3])
    #cmd.save(out_name,pocket)
    cmd.delete('all')


