import os


pdbdir = ''
pocketdir = ''
list = os.listdir(pdbdir)

#num = 1
for pdb_id in list:
    pdb_filename = pdbdir + pdb_id
    out_name = pocketdir +pdb_id
    file = open(pdb_filename, "r")
    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if (line[0:6]) in ("HETATM") and (line[17:20]) not in (''):
            name = line[17:20]
            str(name)
            name.strip()
            break

    #print(pdb_id, name,num)
    #num  = num + 1




    cmd.load(pdb_filename)
     #cmd.select()
    #print(name)

    cmd.select()
    cmd.save(out_name,pdb_id[0:1]+'s')
    cmd.delete('all')


