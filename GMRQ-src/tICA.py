#!/usr/bin/env python
'''
@Author ymh
@Email  maohuay@hotmail.com
@Date   2020-11-18 10:05:20
@Web    https://github.com/Aunity
'''

import sys
import pandas as pd
from msmbuilder.dataset import dataset
from msmbuilder.decomposition import tICA
import warnings

warnings.filterwarnings('ignore')
def main():
    scaled_diheds = dataset('scaled_Heavydist_rmsd')
    tica_model = tICA(lag_time=300, n_components=3, kinetic_mapping=True) # 6ns 3IC, 0.02 ns*300 = 6 ns
    tica_model = scaled_diheds.fit_with(tica_model)
    tica_trajs = scaled_diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')
    pd.to_pickle(tica_model, 'tica_model.pkl3')

if __name__ == '__main__':
    main()
