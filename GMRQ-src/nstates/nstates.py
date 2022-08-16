#!/usr/bin/env python
'''
@Author ymh
@Email  maohuay@hotmail.com
@Date   2020-11-06 15:39:27
@Web    https://github.com/Aunity
'''

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from msmbuilder.dataset import dataset
from msmbuilder.cluster import MiniBatchKMeans,MiniBatchKMedoids, KCenters
from msmbuilder.msm import MarkovStateModel


def model_valid(inp, outf, dt=1):
    nstates = [50, 100, 150, 200, 250, 300, 250, 400, 500, 600, 800, 1000, 1200]
    scaled_diheds = dataset(inp)
    trajectories = [t[::dt,:].copy() for t in scaled_diheds]
    cv = KFold(len(trajectories), n_folds=5)
    results = []
    lagtime = 50/dt
    model = Pipeline([
            ('cluster', KCenters(random_state=43)),
            ('msm', MarkovStateModel(n_timescales=1, lag_time=lagtime, reversible_type='mle', verbose=False)),
            ])

    for nstate in nstates:
        model.set_params(cluster__n_clusters=nstate)
        for fold, (train_index, test_index) in enumerate(cv):
            print(nstate, fold)
            train_data = [trajectories[i] for i in train_index]
            test_data = [trajectories[i] for i in test_index]

            # fit model with a subset of the data (training data).
            # then we'll score it on both this training data (which
            # will give an overly-rosy picture of its performance)
            # and on the test data.
            model.fit(train_data)
            train_score = model.score(train_data)
            test_score = model.score(test_data)

            results.append({
                'train_score': train_score,
                'test_score': test_score,
                'nstate': nstate,
                'fold': fold})

    results = pd.DataFrame(results)
    results.head()

    pd.to_pickle(results, outf)

    avgs = (results
             .groupby('nstate')
             .aggregate(np.median)
             .drop('fold', axis=1))
    print(avgs)

    best_n = avgs['test_score'].argmax()
    best_score = avgs.loc[best_n, 'test_score']
    print(best_n, "gives the best score:", best_score)


def main():
    if len(sys.argv[1:]) != 2:
        print('Usage:python %s <scaled_diheds_fp> <outf>'%sys.argv[0])
        sys.exit(0)
    fp, outf = sys.argv[1:]
    model_valid(fp, outf, dt=1)

if __name__ == '__main__':
    main()
