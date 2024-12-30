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
    #cmd.select(pdb_id[0:3],'byres'+' '+' '+a+' '+'around'+' '+'9'  )
    cmd.createpocket(pocket,pdb_id within 5 of resn a)
    #cmd.save(out_name,pdb_id[0:3])
    cmd.save(out_name,pocket)
    cmd.delete('all')


RNA_pdb_names = os.listdir(rna_pdb_dir)

for pdb_id in RNA_pdb_names:
    pdb_filename = rna_pdb_dir + pdb_id
    file = open(pdb_filename, "r")
    lines = file.readlines()
    id_list = []
    cx = cy = cz = cont = 0
    for i in range(len(lines)):
        line = lines[i]
        if (line[0:6]) in ("HETATM"):
            x = float(line[32:39])
            y = float(line[40:47])
            z = float(line[48:55])
            cx = cx + x
            cy = cy + y
            cz = cz + z
            cont = cont + 1
            p = (cx, cy, cz)
    p = (cx/cont, cy/cont, cz/cont)

    for j in range(len(lines)):
        line = lines[j]
        if (line[0:4]) in ("ATOM"):
            a = float(line[32:39])
            b = float(line[40:47])
            c = float(line[48:55])
            q = (a, b, c)
            dis = math.dist(p, q)
            if dis < 12:
                id_list.append(j)




    finished_id = list(set(id_list))
    newfile = open(pocket_dir + pdb_id, "w")
    for temp in range(len(finished_id)):
        line = lines[finished_id[temp]]
        #print(line)
        newfile.write(line)
    print(pocket_dir + pdb_id,"ok")
