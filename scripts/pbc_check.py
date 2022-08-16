#!/usr/bin/env python
import os
import sys
import copy
import mdtraj as md
import numpy as np
from numpy.linalg import norm
from mdtraj.geometry._geometry import _find_closest_contact

def obtain_pbc(xtc, axis=0, outfile=None):
    '''
    axis: 0  1  2, x y z
    '''
    traj = copy.deepcopy(xtc)
    xyz = xtc.xyz
    boxvectors = traj.unitcell_vectors
    for i in range(traj.n_frames):
        traj.xyz[i] += boxvectors[i, axis]
    if outfile:
        traj[-1].save(outfile)
    return traj

def minimal_distance_pbc(reftraj, traj, axis=0, selstr='protein', outfile=None):
    distance = []
    index = reftraj.top.select(selstr)
    reftrajSelected = reftraj.atom_slice(index)
    trajSelected = traj.atom_slice(index)
    atoms1 = np.arange(reftrajSelected.n_atoms, dtype=np.int32)
    atoms2 = np.arange(reftrajSelected.n_atoms, dtype=np.int32)+len(atoms1)

    for i in range(traj.n_frames):
        coordi = reftrajSelected.xyz[i]
        coordj = trajSelected.xyz[i]
        box = reftrajSelected.unitcell_vectors[i]
        xyz = np.r_[coordi, coordj]
        atom1, atom2, dist = _find_closest_contact(xyz, atoms1, atoms2, box=None)
        distance.append(dist)
    return distance

def check_pbc(trajfile, top):
    xtc = md.load(trajfile, top=top)
    xyz = copy(xtc.xyz)


def main():
    if len(sys.argv[1:])!=3:
        print('Usage: python %s <trajfile> <topfile> <axis>'%sys.argv[0])
        exit(0)
    trajfile, top, axis  = sys.argv[1:]
    xtc = md.load(trajfile, top=top)
    #xtc[-1].save('test-ref.pdb')
    axis = 2
    traj = obtain_pbc(xtc, axis=axis)#, outfile='test-z.pdb')
    distance = np.array(minimal_distance_pbc(xtc, traj, axis))
    print('Distance statistic:')
    print('Mean: %.4f'%distance.mean())
    print('Range: %.4f ~ %.4f'%(distance.min(), distance.max()))
    print('Min distance frameID: %d'%(distance.argmin()))
    np.savetxt('pbc-check.txt', distance, fmt='%.3f')

if __name__ == "__main__":
    main()
