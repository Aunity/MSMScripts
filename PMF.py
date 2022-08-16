import os
import sys
import numpy as  np
import pandas as pd

from msmbuilder.dataset import dataset
import pyemma

import matplotlib.pyplot as plt


ticaspath, dtrajfile = sys.argv[1:]
#jobdir = '/home/maohua/MY1/amber14sb'
lag = 1500
ticas = dataset(ticaspath)

txx = np.concatenate(ticas)
print(txx.shape)
dtrajs = pd.read_pickle(dtrajfile)
msm = pyemma.msm.bayesian_markov_model(dtrajs, lag=lag, show_progress=False)

w = np.concatenate(msm.trajectory_weights())

fig, axs = plt.subplots(ncols=2, figsize=(12,5))
ax = axs[0]
pyemma.plots.plot_free_energy(*txx[:,[0,1]].T, ax=ax)
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
ax.set_title('MD simulation')

ax=axs[1]
pyemma.plots.plot_free_energy(*txx[:,[0,1]].T, ax=ax, weights=w)
ax.set_xlabel('IC 1')
ax.set_ylabel('IC 2')
ax.set_title('MSM ensemble')
fig.tight_layout()
fig.savefig('PMF.png', dpi=300)
