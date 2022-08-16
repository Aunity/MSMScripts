#!/usr/bin/env python
'''
@Author ymh
@Email  maohuay@hotmail.com
@Date   2021-03-07 04:09:59
@Web    https://github.com/Aunity
'''

import os,sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
#def native_atom_pair(native):
#    from itertools import combinations
#    BETA_CONST = 50  # 1/nm
#    LAMBDA_CONST = 1.8
#    NATIVE_CUTOFF = 0.45  # nanometers
#
#    # get the indices of all of the heavy atoms
#    MDM2 = native.atom_slice(native.top.select('(resid 0 to 86)'))
#    heavy0 = MDM2.topology.select_atom_indices('heavy')
#    p53 = native.atom_slice(native.top.select('(resid 87 to 103)'))
#    heavy1 = p53.topology.select_atom_indices('heavy') + MDM2.top.n_atoms
#    # get the pairs of heavy atoms which are farther than 3
#    # residues apart
#    heavy_pairs = np.array([(i,j) for i in heavy0 for j in heavy1 \
#                            if abs(native.topology.atom(i).residue.index - \
#                            native.topology.atom(j).residue.index) > 3])
#
#    # compute the distances between these pairs in the native state
#    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
#    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
#    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
#    print("Number of native contacts", len(native_contacts))
#
#    return native_contacts
#'''
#native Heavy atom distance
#'''

#from msmbuilder.featurizer import AtomPairsFeaturizer
from msmbuilder.dataset import dataset
import mdtraj as md
from msmbuilder.preprocessing import RobustScaler

if len(sys.argv[1:])!=2:
    print('Usage: python %s <top> <XTC_DIR>'%sys.argv[0])
    exit(0)
top, xtcp = sys.argv[1:]
xyz = dataset("%s/*.xtc"%xtcp, topology=top, stride=1)

refpdb = md.load_pdb(top)

from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

print(xyz[0].xyz.shape)
print(diheds[0].shape)

from msmbuilder.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_diheds = diheds.fit_transform_with(scaler, 'scaled_diheds/', fmt='dir-npy')

print(diheds[0].shape)
print(scaled_diheds[0].shape)

# from itertools import combinations
# atom_pair = native_atom_pair(refpdb)
#featurizer = AtomPairsFeaturizer(atom_pair)

#diheds = xyz.fit_transform_with(featurizer, 'Heavydist/', fmt='dir-npy')
pd.to_pickle(featurizer, 'featuer-model.pkl3')

# scaler = RobustScaler()
# scaled_diheds = diheds.fit_transform_with(scaler, 'scaled_Heavydist', fmt='dir-npy')
pd.to_pickle(scaler, 'scaler-model.pkl3')
