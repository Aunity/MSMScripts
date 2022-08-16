import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import pyemma

def run_timescales(dtrajfile):
    dtrajs = pd.read_pickle(dtrajfile)
    lags = [1]
    lags.extend(np.arange(100, 10001,100))  # 依据5000为单条轨迹的frame长度
    its = pyemma.msm.timescales_msm(dtrajs, lags=lags, nits=25)
    dt = 0.02 # units: ns  需要依据自己的轨迹情况进行修改
    units = 'ns'
    fig, ax = plt.subplots()
    pyemma.plots.plot_implied_timescales(its, ax=ax, units=units, dt=dt, linewidth=3)
    fig.tight_layout()
    fig.savefig('timescale.png', dpi=150)
    pd.to_pickle(its, 'its.pkl3')

def main():
    dtrajfile = sys.argv[1]
    run_timescales(dtrajfile)

if __name__ == "__main__":
    main()
