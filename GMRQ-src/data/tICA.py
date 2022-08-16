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
    scaled_diheds = dataset('scaled_Heavydist_1ns')
    tica_model = pd.read_pickle('tica_model.pkl3') 
    tica_trajs = scaled_diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

if __name__ == '__main__':
    main()
