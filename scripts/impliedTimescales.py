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
    lags.extend(np.arange(100, 5001,100))
    its = pyemma.msm.timescales_msm(dtrajs, lags=lags, nits=25)
    dt = 0.02 # units: ns
    units = 'ns'
    fig, ax = plt.subplots()
    pyemma.plots.plot_implied_timescales(its, ax=ax, units=units, dt=dt, linewidth=3)
    fig.tight_layout()

def main():
    main()

if __name__ == "__main__":
    main()