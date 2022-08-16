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
from msmbuilder.featurizer import AtomPairsFeaturizer
from msmbuilder.dataset import dataset
import mdtraj as md
from msmbuilder.preprocessing import RobustScaler,MinMaxScaler

# top = "../p53-complex-c36.pdb"
# xyz = dataset("../xtc_meta/*.xtc",topology=top,stride=1)

# refpdb = md.load_pdb(top)
from itertools import combinations
diheds =  dataset('Heavydist_rmsd/')

# dt = 0.02 ns * 50 = 1ns


scaler = MinMaxScaler()
scaled_diheds = diheds.fit_transform_with(scaler, 'scaled_Heavydist_rmsd', fmt='dir-npy')
pd.to_pickle(scaler, 'scaler-model.pkl3')

#scaler = pd.read_pickle('./scaler-model-dt1ns.pkl3')
#diheds = dataset('./Heavydist_rmsd/')

#scaled_diheds = diheds.transform_with(scaler,'scaled_Heavydist_rmsd', fmt='dir-npy')
