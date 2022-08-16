import pandas as pd
from msmbuilder.tpt import hub_scores
M = pd.read_pickle('M_lag1500.pkl3')
score = hub_scores(M)

pd.to_pickle(score, 'score.pkl3')
