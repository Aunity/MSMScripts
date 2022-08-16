import sys
import pandas as pd
from msmbuilder.msm import MarkovStateModel

dtraj = pd.read_pickle(sys.argv[1])
lagtime = 1500
M = MarkovStateModel(lag_time=lagtime) #其他参数默认
M.fit(dtraj)
pd.to_pickle(M, 'M_lag1500.pkl3')
