from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib
from my_hmm import create_markov_model
#plot
import csgo_plot
import time


def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    _dim = 1 if len(a.shape) < 2 else a.shape[1]
    if _dim == 1:
    	return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )
    else:
    	length = n 
    	remainder = n % width 
    	result = np.stack(a[i:width+i] for i in range(0,length-width))
    	if remainder == 0:
    		return result 
    	else:
    		# gets the remainder and stacks it.
    		## Todo: sort out missing data.
    		#return np.vstack([result, a[-remainder:]])
    		return result


def sliding_window_predict(test_data,model_fair, model_hacker, window_size=128):
	windows = window_stack(test_data, width=window_size)

	# Uh... Don't like how this looks like as it can error.
	_dim = 1 if len(windows[0].shape) < 2 else windows[0].shape[1]

	# Scores should be array X by 1 Dimensions
	scores_fair = np.empty(( len(windows) , 1), float)
	scores_hacker = np.empty(( len(windows) , 1), float)

	for i, window in enumerate(windows):
		scores_fair[i] =   model_fair.score(window.reshape(len(window),  _dim))
		scores_hacker[i] = model_hacker.score(window.reshape(len(window), _dim))

	return scores_fair, scores_hacker


def classify_analysis(dfhacker, dffair, columns, n_components=4):
	model_h, X_h, test_round_h = create_markov_model(dfhacker, columns, n_components=n_components, test_round=6)

	csgo_plot.plot_plane_hmm(np.arange(len(X_h)), X_h, model_h, X_h, title="n-"+str(n_components)+"-".join(columns)+" -- Hacker")

	plt.show(block=False)

	#####
	## Fair

	model_f, X_f, test_round_f = create_markov_model(dffair , columns, n_components=n_components, test_round='r') 

	csgo_plot.plot_plane_hmm(np.arange(len(X_f)), X_f, model_f, X_f, title="n-"+str(n_components)+"-".join(columns)+" -- Fair")

	plt.show(block=False)

	test_hacker = test_round_h[0][columns].as_matrix()
	_dim = 1 if len(test_hacker.shape) < 2 else test_hacker.shape[1]
	test_hacker = test_hacker.reshape(len(test_hacker),_dim)

	test_fair = test_round_f[0][columns].as_matrix()
	_dim = 1 if len(test_fair.shape) < 2 else test_fair.shape[1]
	test_fair = test_fair.reshape(len(test_fair), _dim)

	scores_fair, scores_hacker = sliding_window_predict(test_hacker, model_fair=model_f, model_hacker=model_h)

	# Meaning positive scores mean the hacker is classified
	# negative scores mean the its the fair player.
	diff_scores = scores_hacker - scores_fair

	length_diff = len(test_hacker) - len(diff_scores)
	#

	## Should be 1 dimensoional as were simply stacking 0s on the score
	y = np.vstack([np.zeros((length_diff, 1)), diff_scores ])

	# Dont panic just reshaping stuff
	x = test_round_h[0]["Tick"].as_matrix()
	x = x.reshape(len(x), 1)

	csgo_plot.plot_plane_diff(x, y, title="n-"+str(n_components)+"_".join(columns))

	pdb.set_trace()
	print "Stop"



## Hacker / Fair
hackerargs  = {'id':76561197979652439 ,'class':'hacker', 'start_tick':0 , 'end_tick':1000000}

dfhacker = clean_data_to_numbers("32t_de_dust2_hack_2.csv", "32t_de_dust2_hack_2_attackinfo.csv", dictargs=hackerargs)

fairargs = {'id':76561197979669175, 'class':'fair', 'start_tick':60000, 'end_tick':110000}

dffair = clean_data_to_numbers("128t_de_inferno_186_envyus-dignitas_de_inferno.csv", "128t_de_inferno_186_envyus-dignitas_de_inferno_attackinfo.csv", dictargs=fairargs)

classify_analysis(dfhacker, dffair, ["TrueViewDiffSpeed","TrueViewRadDiff"], n_components=12)
# classify_analysis(dfhacker, dffair, ["TrueViewRadDiff"])







# Plot X / Y corelation of score...

pdb.set_trace()
print "stop"





