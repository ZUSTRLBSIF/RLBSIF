Calculate the surface charge.
from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree



def computeCharges(names):
    charge = []
    for name in names:
        charge.append(float(name.split("_")[5]))



    return np.array(charge)

def assignChargesToNewMesh(new_vertices, old_vertices, old_charges, seeder_opts):
    dataset = old_vertices
    testset = new_vertices
    new_charges = np.zeros(len(new_vertices))
    if seeder_opts["feature_interpolation"]:
        num_inter = 4  # Number of interpolation features
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset, k=num_inter)
        # Square the distances (as in the original pyflann)
        dists = np.square(dists)
        for vi_new in range(len(result)):
            vi_old = result[vi_new]
            dist_old = dists[vi_new]
            if dist_old[0] == 0.0:
                new_charges[vi_new] = old_charges[vi_old[0]]
                continue

            total_dist = np.sum(1 / dist_old)
            for i in range(num_inter):
                new_charges[vi_new] += (
                        old_charges[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )
    else:
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset)
        new_charges = old_charges[result]
    return new_charges







