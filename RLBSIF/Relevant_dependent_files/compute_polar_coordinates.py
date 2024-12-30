#Construct the polar coordinate system for surveying.

import sys
from sklearn.manifold import MDS
import networkx as nx
import numpy as np
import scipy.linalg
from IPython.core.debugger import set_trace
from  numpy.linalg import norm
import time
from scipy.sparse import csr_matrix, coo_matrix
import pymesh

def compute_polar_coordinates(mesh, do_fast=True, radius=12, max_vertices=200):


 
    vertices = mesh.vertices
    faces = mesh.faces
    norm1 = mesh.get_attribute('vertex_nx')
    norm2 = mesh.get_attribute('vertex_ny')
    norm3 = mesh.get_attribute('vertex_nz')
    normals = np.vstack([norm1, norm2, norm3]).T


    G=nx.Graph()
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))


    f = np.array(mesh.faces, dtype = int)
    rowi = np.concatenate([f[:,0], f[:,0], f[:,1], f[:,1], f[:,2], f[:,2]], axis = 0)
    rowj = np.concatenate([f[:,1], f[:,2], f[:,0], f[:,2], f[:,0], f[:,1]], axis = 0)
    edges = np.stack([rowi, rowj]).T
    verts = mesh.vertices


    edgew = verts[rowi] - verts[rowj]
    edgew = scipy.linalg.norm(edgew, axis=1)
    wedges = np.stack([rowi, rowj, edgew]).T

    G.add_weighted_edges_from(wedges)
    start = time.clock()
    if do_fast:
        dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius)
    else:
        dists = nx.all_pairs_dijkstra_path_length(G, cutoff=radius*2)
    d2 = {}
    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.clock()
    print('Dijkstra took {:.2f}s'.format((end-start)))
    D = dict_to_sparse(d2)


    idx = {}
    for ix, face in enumerate(mesh.faces):
        for i in range(3):
            if face[i] not in idx:
                idx[face[i]] = []
            idx[face[i]].append(ix)


    i = np.arange(D.shape[0])
 
    D[i,i] = 1e-8

    mds_start_t = time.clock()

    if do_fast:
        theta = compute_theta_all_fast(D, vertices, faces, normals, idx, radius)
    else:
        theta = compute_theta_all(D, vertices, faces, normals, idx, radius)
    

    mds_end_t = time.clock()
    print('MDS took {:.2f}s'.format((mds_end_t-mds_start_t)))
    
    n = len(d2)
    theta_out = np.zeros((n, max_vertices))
    rho_out= np.zeros((n, max_vertices))
    mask_out = np.zeros((n, max_vertices))

    neigh_indices = []
    

    for i in range(n): 
        dists_i = d2[i]
        sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])
        neigh = [int(x[0]) for x in sorted_dists_i[0:max_vertices]]
        neigh_indices.append(neigh)
        rho_out[i,:len(neigh)]= np.squeeze(np.asarray(D[i,neigh].todense()))
        theta_out[i,:len(neigh)]= np.squeeze(theta[i][neigh])
        mask_out[i,:len(neigh)] = 1
   
    theta_out[theta_out < 0] +=2 * np.pi



def compute_thetas(plane, vix, verts, faces, normal, neighbors, idx):

    plane_center_ix = np.where(neighbors == vix)[0][0]
    thetas = np.zeros(len(verts))

    plane = plane-plane[plane_center_ix]


    valid = False
    for i in range(len(idx[vix])):
        tt = idx[vix][i]
        tt = faces[tt]
 
        check_valid = [x for x in tt if x in neighbors]
        if len(check_valid) == 3:
            valid = True
            break
    try:
        assert(valid)
    except:
        set_trace()

    normal_tt = np.mean([normal[tt[0]], normal[tt[1]], normal[tt[2]]], axis=0)


    neigh_tt = [x for x in tt if x != vix]
    v1ix = neigh_tt[0]
    v2ix = neigh_tt[1]

    v1ix_plane = np.where(neighbors == v1ix)[0][0]
    v2ix_plane = np.where(neighbors == v2ix)[0][0]


    norm_plane = np.sqrt(np.sum(np.square(plane),axis=1))

    norm_plane[plane_center_ix] = 1.0
    norm_plane = np.stack([norm_plane, norm_plane], axis=1)

    vecs = np.divide(plane,norm_plane)
    vecs[plane_center_ix] = [0,0]
    vecs = np.stack([vecs[:,0], vecs[:,1], np.zeros(len(vecs))], axis=1)

    ref_vec = vecs[v1ix_plane]


    term1 = np.sqrt(np.sum(np.square(np.cross(vecs,ref_vec)),axis=1))
    term1 = np.arctan2(term1, np.dot(vecs,ref_vec))
    normal_plane = [0.0,0.0,1.0]
    theta = np.multiply(term1, np.sign(np.dot(vecs,np.cross(normal_plane,ref_vec))))

    v0 = verts[vix]
    v1 = verts[v1ix]
    v2 = verts[v2ix]
    v1 = v1 - v0
    v1 = v1/np.linalg.norm(v1)
    v2 = v2 - v0
    v2 = v2/np.linalg.norm(v2)
    angle_v1_v2 = np.arctan2(norm(np.cross(v2,v1)),np.dot(v2,v1))*np.sign(np.dot(v2,np.cross(normal_tt,v1)))

    sign_3d = np.sign(angle_v1_v2)
    sign_2d = np.sign(theta[v2ix_plane])
    if sign_3d != sign_2d:
        theta = -theta
    theta[theta == 0] = np.finfo(float).eps
    thetas[neighbors] = theta

    return thetas

def dict_to_sparse(mydict):


    data = []
    row = []
    col = []
    for r in mydict.keys():
        for c in mydict[r].keys():
            r = int(r)
            c = int(c)
            v = mydict[r][c]
            data.append(v)
            row.append(r)
            col.append(c)
    coo = coo_matrix((data,(row,col)))
    return csr_matrix(coo)




def extract_patch(mesh, neigh, cv):

    n = len(mesh.vertices)
    subverts = mesh.vertices[neigh]

    nx = mesh.get_attribute('vertex_nx')
    ny = mesh.get_attribute('vertex_ny')
    nz = mesh.get_attribute('vertex_nz')
    normals = np.vstack([nx, ny, nz]).T
    subn = normals[neigh]


    
    m = np.zeros(n,dtype=int)


    m = m - 1 
    for i in range(len(neigh)):
        m[neigh[i]] = i
    f = mesh.faces.astype(int)
    nf = len(f)
    
    neigh = set(neigh) 
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf) \
             if f[i][0] in neigh and f[i][1] in neigh and f[i][2] in neigh]
    
    subfaces = subf
    return np.array(subverts), np.array(subn), np.array(subf) 

def output_patch_coords(subv, subf, subn, i, neigh_i, theta, rho): 
      
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)

    rho = np.array([rho[0,ix] for ix in range(rho.shape[1]) if ix in neigh_i])
    mesh.add_attribute('rho')
    mesh.set_attribute('rho', rho)

    theta= np.array([theta[ix] for ix in range((theta.shape[0])) if ix in neigh_i])
    mesh.add_attribute('theta')
    mesh.set_attribute('theta', theta)

    charge = np.zeros(len(neigh_i))
    mesh.add_attribute('charge')
    mesh.set_attribute('charge', charge)

    pymesh.save_mesh('v{}.ply'.format(i), mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)


def call_mds(mds_obj, pair_dist):
    return mds_obj.fit_transform(pair_dist)

def compute_theta_all(D, vertices, faces, normals, idx, radius):
    mymds = MDS(n_components=2, n_init=1, max_iter=50, dissimilarity='precomputed', n_jobs=10)
    all_theta = []
    for i in range(D.shape[0]):
        if i % 100 == 0:
            print(i)

        neigh = D[i].nonzero()
        ii = np.where(D[i][neigh] < radius)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = D[neigh_i,:][:,neigh_i]
        pair_dist_i = pair_dist_i.todense()


        plane_i = call_mds(mymds, pair_dist_i)

        theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
        all_theta.append(theta)
    return all_theta


def compute_theta_all_fast(D, vertices, faces, normals, idx, radius):

    mymds = MDS(n_components=2, n_init=1, eps=0.1, max_iter=50, dissimilarity='precomputed', n_jobs=1)
    all_theta = []
    start_loop = time.clock()
    for i in range(D.shape[0]):

        neigh = D[i].nonzero()

        ii = np.where(D[i][neigh] < radius/2)[1]
        neigh_i = neigh[1][ii]
        pair_dist_i = D[neigh_i,:][:,neigh_i]
        pair_dist_i = pair_dist_i.todense()

    
        tic = time.clock()
        plane_i = call_mds(mymds, pair_dist_i)
        toc = time.clock()
        only_mds += (toc - tic)
    
  
        theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)


        kk = np.where(D[i][neigh] >= radius/2)[1]
        neigh_k = neigh[1][kk]
        dist_kk = D[neigh_k,:][:,neigh_i]
        dist_kk = dist_kk.todense()
        dist_kk[dist_kk == 0] = float('inf')
        closest = np.argmin(dist_kk, axis=1)
        closest = np.squeeze(closest)
        closest = neigh_i[closest]
        theta[neigh_k] = theta[closest]

        
        all_theta.append(theta)
    end_loop = time.clock()
    print('Only MDS time: {:.2f}s'.format(only_mds))
    print('Full loop time: {:.2f}s'.format(end_loop-start_loop))






