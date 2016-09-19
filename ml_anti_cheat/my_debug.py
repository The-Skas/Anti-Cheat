"""
Used as a temporary debug file to load functionality without restarting the program
"""



# The + 1.0 / - 1.0 here prevent a 0 denominator, preventing an error.
from plot import csgo_plot
def result_probability(hacker, fair):
	if( (fair < 0) & (hacker > 0)):
		hacker = (hacker - fair) + 1.0
		fair = 1.0
		#do normalize
	elif( (fair >0)  & (hacker < 0)):
		fair = (fair - hacker) +  1.0
		hacker = 1.0
		# do normalize
	elif( (fair < 0) & (hacker < 0)):
		# always has a ratio  < 1 and > 0?
		temp = fair
		fair = hacker - 1.0
		hacker = temp - 1.0

	ratio = hacker / fair
	if( ratio >= 1):
		return 1.0 - 1.0 / ratio 
	else:
		return (ratio - 1.0)

import numpy as np
from my_hmm import create_markov_model
import matplotlib.pyplot as plt
from hmm_classify import sliding_window_predict
def classify_analysis(dfhacker, dffair, columns, n_components=4, window_size=128):
	# Hacker
	model_h, X_h, test_round_h = create_markov_model(dfhacker, columns, n_components=n_components, test_round=6)
	hack_round = test_round_h[0].iloc[0].Round

	csgo_plot.plot_plane_hmm(np.arange(len(X_h)), X_h, model_h, X_h, title="Hack-R-"+str(hack_round)+"n-"+str(n_components)+"-".join(columns)+" -- Hacker")

	plt.show(block=False)

	#####
	## Fair

	model_f, X_f, test_round_f = create_markov_model(dffair , columns, n_components=n_components, test_round='r') 
	fair_round = test_round_f[0].iloc[0].Round

	csgo_plot.plot_plane_hmm(np.arange(len(X_f)), X_f, model_f, X_f, title="Fair-R-"+str(fair_round)+"n-"+str(n_components)+"-".join(columns)+" -- Fair")

	plt.show(block=False)

	for name, test_round in zip(["Hacker-R"+str(hack_round),"Fair-R"+str(fair_round)],[test_round_h, test_round_f]):
		test_data = test_round[0][columns].as_matrix()
		_dim = 1 if len(test_data.shape) < 2 else test_data.shape[1]
		test_data = test_data.reshape(len(test_data),_dim)

		scores_fair, scores_hacker = sliding_window_predict(test_data, model_fair=model_f, model_hacker=model_h, window_size=window_size)

		# Meaning positive scores mean the hacker is classified
		# negative scores mean the its the fair player.
		diff_scores = scores_hacker - scores_fair

		length_diff = len(test_data) - len(diff_scores)
		#

		## Should be 1 dimensoional as were simply stacking 0s on the score
		y = np.vstack([np.zeros((length_diff, 1)), diff_scores ])

		# Dont panic just reshaping stuff
		x = test_round[0]["Tick"].as_matrix()
		x = x.reshape(len(x), 1)

		csgo_plot.plot_plane_diff(x, y, title=name+"n-"+str(n_components)+"_".join(columns))

		# Calculate probability and plot
		func = np.vectorize(result_probability)
		probability_scores = func(scores_hacker, scores_fair)
		y = np.vstack([np.zeros((length_diff, 1)), probability_scores ])
		csgo_plot.plot_plane_diff(x, y, title=name+" Probability n-"+str(n_components)+"_".join(columns))


		pdb.set_trace()
		print "Stop"

def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    _dim = 1 if len(a.shape) < 2 else a.shape[1]
    if _dim == 1:
    	return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )
    else:
    	length = n 
    	width = width if width < length else length-1
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
