import os
import sys
import pyemma
import numpy as np
import pandas as pd
from msmbuilder.cluster import KCenters, MiniBatchKMeans, MiniBatchKMedoids
nclusters = [500, 800, 1000, 2000]

inputfile = sys.argv[1]
trajs_tica = pd.read_pickle(inputfile)
methods = {
    'kcenters': KCenters,
    #'minibatchkmeans': MiniBatchKMeans,
    'minibatchkmedoids': MiniBatchKMedoids
}
seed = 43
#try:
if 1:
    for name, method in methods.items():
        for n in nclusters:
            jobname = '%s-%d'%(name, n)
            outp = os.path.join(name, 'n%d'%n)
            if not os.path.exists(outp):
                os.system('mkdir -p %s'%outp)       
            cluster = method(n_clusters=n, random_state=seed)
            dtrajs = cluster.fit_transform(trajs_tica)
            f1 = os.path.join(outp, 'cluster.pkl3')
            f2 = os.path.join(outp, 'dtrajs.pkl3')
            pd.to_pickle(cluster, f1)
            pd.to_pickle(dtrajs, f2)
            print(outp)
#except:
#    print('error run %s'%outp)

outp = 'emmaKmeans'
if not os.path.exists(outp):
    os.mkdir(outp)
for n in nclusters:
    cluster = pyemma.coordinates.cluster_kmeans(data=trajs_tica, k=n)
    dtrajs = [s.T[0] for s in cluster.get_output()]

    f1 = os.path.join(outp, 'cluster.pkl3')
    f2 = os.path.join(outp, 'dtrajs.pkl3')
    pd.to_pickle(cluster, f1)
    pd.to_pickle(dtrajs, f2)