
from default_config.chemistry import radii

def output_pdb_as_xyzrn(pdbfilename, xyzrnfilename, Mg = None):
    fid = open(pdbfilename, "r")
    lines = fid.readlines()
    outfile = open(xyzrnfilename, "w")
    coords = None
    Mg_coords = []

    for i in range(len(lines) - 1):
        line = lines[i]
        atom_ix = line[7:11].strip()
        atom_x = float(line[30:38].strip())
        atom_y = float(line[39:46].strip())
        atom_z = float(line[48:56].strip())
        chain_id = line[21].strip()
        chain_ix = line[24:26].strip()
        chain_res = line[19].strip()
        chain_res_atom = line[12:16].strip()
        atomtype = chain_res_atom[0].strip()
        charge = line[70:76].strip()
        if atomtype == "M":
            Mg_coords = [atom_x, atom_y, atom_z]

        if Mg == True:  #Note: Mg ions are not included in our dataset.


            if atomtype == "M":
                coords = "{:.06f} {:.06f} {:.06f}".format(
                    atom_x, atom_y, atom_z)
                full_id = "{}_{}_{}_{}_{}_{}".format(
                    chain_id, chain_ix, atom_ix, chain_res, chain_res_atom, charge
                )
                rad = radii[atomtype]
        else:
            # ignore Mg
            if atomtype == "M":
                continue
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom_x, atom_y, atom_z)
            full_id = "{}_{}_{}_{}_{}_{}".format(
                chain_id, chain_ix, atom_ix, chain_res, chain_res_atom, charge
            )
            rad = radii[atomtype]
        if coords is not None:
            outfile.write(coords + " " + rad + " 1 " + full_id + "\n")
    return Mg_coords
