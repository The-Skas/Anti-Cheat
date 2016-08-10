from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib
import my_hmm
#plot
import csgo_plot
import time

## Hacker
hackerargs  = {'id'=76561197979652439 ,'class'=hacker, 'start_tick'=0 , 'end_tick':1000000}

dfhacker = clean_data_to_numbers("32t_de_dust2_hack_2.csv", "32t_de_dust2_hack_2_attackinfo.csv", dictargs=hackerargs)

model_h, X_h, test_round_h = create_markov_model(dfhacker, ["TrueViewDiffSpeed"], n_components=4, test_round=6)

csgo_plot.plot_plane(np.arange(len(X_h)), X_h, model_h, X_h, title=dfhacker["class"]+" -- TrueViewRadDiff")

plt.show(block=False)

## Fair
fairargs = {'id':76561197979669175, 'class':fair, 'start_tick':60000, 'end_tick':110000}

dffair = clean_data_to_numbers("128t_de_inferno_186_envyus-dignitas_de_inferno.csv", "128t_de_inferno_186_envyus-dignitas_de_inferno_attackinfo.csv", dictargs=fairargs)

model_f, X_f, test_round_f = create_markov_model(dffair , ["TrueViewDiffSpeed"], n_components=4, test_round='r') 

csgo_plot.plot_plane(np.arange(len(X_f)), X_f, model_f, X_f, title=fairargs["class"]+" -- TrueViewRadDiff")

plt.show(block=False)

pdb.set_trace()





