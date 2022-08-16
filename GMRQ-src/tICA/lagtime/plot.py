#!/usr/bin/env python
'''
@Author ymh
@Email  maohuay@hotmail.com
@Date   2020-11-06 15:53:03
@Web    https://github.com/Aunity
'''
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('/home/ymh/mybin/ymh.mplstyle')

def plot(results, outf, paras):
    lags = results[paras]
    avgs = (results
           .groupby(paras)
           .aggregate(np.median)
           .drop('fold', axis=1))
    best_n = avgs['test_score'].argmax()
    best_score = avgs.loc[best_n, 'test_score']
    print(best_n, "%s gives the best score:"%paras, best_score)

    fig, ax = plt.subplots()
    #ax.scatter(results[paras], results['train_score'], c='b', lw=0, label=None)
    #ax.scatter(results[paras], results['test_score'],  c='r', lw=0, label=None)

    #ax.plot(avgs.index, avgs['test_score'], c='r', lw=2, label='Mean test')
    #ax.plot(avgs.index, avgs['train_score'], c='b', lw=2, label='Mean train')

    test_error = results.groupby(paras).std()['test_score']
    train_error = results.groupby(paras).std()['train_score']
    ax.errorbar(avgs.index, avgs['test_score'], yerr=test_error, color='red', ecolor='red',capsize=5, markeredgewidth=1)
    ax.errorbar(avgs.index, avgs['train_score'], yerr=train_error, color='blue', ecolor='blue', capsize=5, markeredgewidth=1)

    #ax.plot(best_n, best_score, c='r', marker='*', ms=20, label='{} {}'.format(best_n, paras))

    #ax.semilogy(0.5)
    #ax.set_xlim((min(lags)*.5, max(lags)*5))
    #ax.set_ylim(1.6,2.2)
    ax.set_ylabel('GMRQ')
    ax.set_xlabel(paras)

    #fig.legend(loc='lower right', numpoints=1)
    fig.tight_layout()
    fig.savefig(outf, dpi=300)


def main():
    if len(sys.argv[1:]) != 2:
        print('Usage:python %s <inf> <paras>'%sys.argv[0])
        sys.exit(0)
    inf, paras = sys.argv[1], sys.argv[2]
    outf = os.path.split(inf)[-1]+".png"
    results = pd.read_pickle(inf)
    plot(results, outf, paras)
if __name__ == '__main__':
    main()
