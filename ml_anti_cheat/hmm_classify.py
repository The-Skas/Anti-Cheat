from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib
from my_hmm import create_markov_model, predict_rounds_markov_model
#plot
from plot import csgo_plot
import time


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

import my_debug
# The + 1.0 / - 1.0 here prevent a 0 denominator, preventing an error.
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


def classify_analysis(dfhacker, dffair, columns, n_components=4, window_size=128, dictrounds_hacker=None, dictrounds_fair=None, drawplot=True, custom_test_data = None):
	# Hacker
	import gc
	gc.collect()
	from sklearn.preprocessing import RobustScaler
	robust_scaler = RobustScaler()
	
	test_round_arg = "r" if custom_test_data is None else None																							   #test_round=6
	model_h, X_h, test_round_h  = create_markov_model(dfhacker, columns, n_components=n_components, test_round=test_round_arg, normalizer=None)
	# Given rounds and new HMM .. #fit new model to that

	hack_round = test_round_h[0].iloc[0].Round if custom_test_data is None else "-CUSTOM-"

	csgo_plot.plot_plane_hmm(np.arange(len(X_h)), X_h, model_h, X_h, title="Hack-R-"+str(hack_round)+"n-"+str(n_components)+"-".join(columns)+" -- Hacker")

	plt.show(block=False)

	#####
	## fairargs
																						 #test_round=7
	model_f, X_f, test_round_f = create_markov_model(dffair , columns, n_components=n_components, test_round=test_round_arg, dictrounds=dictrounds_fair, normalizer=None) 
	fair_round = test_round_f[0].iloc[0].Round if custom_test_data is None else "-CUSTOM-"

	csgo_plot.plot_plane_hmm(np.arange(len(X_f)), X_f, model_f, X_f, title="Fair-R-"+str(fair_round)+"n-"+str(n_components)+"-".join(columns)+" -- Fair")

	plt.show(block=False)

	TIME_C = max(test_round_f[0].iloc[0].TimeDiff, test_round_h[0].iloc[0].TimeDiff ) if custom_test_data is None else custom_test_data.dropna().iloc[0].TimeDiff
	if(drawplot):
		for name, test_round in zip(["Hacker-R"+str(hack_round),"Fair-R"+str(fair_round)],[test_round_h, test_round_f]):
			# Needs some refactoring: This basically drops the custom_test_data, which i pass in here... eeeeeeep
			test_round = test_round if custom_test_data is None else (custom_test_data.dropna() , 666)
			test_data = test_round[0][columns].as_matrix() 
	
			_dim = 1 if len(test_data.shape) < 2 else test_data.shape[1]
			test_data = test_data.reshape(len(test_data),_dim)
			# incase normalizing is used otherwise dont
			#test_data = robust_scaler.fit_transform(test_data.reshape(len(test_data),_dim))

			# The warped window size takes into account differences in tick-rates
			# between demos.
			warped_window_size = int(window_size * TIME_C / test_round[0].iloc[0].TimeDiff)
			pdb.set_trace()
			scores_fair, scores_hacker = sliding_window_predict(test_data, model_fair=model_f, model_hacker=model_h, window_size=warped_window_size)

			name = "WinSize-"+str(warped_window_size)+name
			# Meaning positive scores mean the hacker is classified
			# negative scores mean the its the fair player.
			diff_scores = scores_hacker - scores_fair
			diff_scores = np.clip(diff_scores,-2000, 2000)
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


			print "Stop"
	



	dfhacker_new = predict_rounds_markov_model(model=model_h, columns=columns, df=dfhacker)
	dffair_new   = predict_rounds_markov_model(model=model_f, columns=columns, df=dffair)
	pdb.set_trace()
	return dfhacker_new, dffair_new


def classify_analysis_ensemble(dfhacker, dffair, columns):
	#Given two column values.. 
	models = {}
	
	for data, _class in zip([dfhacker,dffair],["hacker","fair"]):
		models[_class] = {}
		for column in columns:
			#generate a model for it.. of 8 components
			models[_class][column] = create_markov_model(data , column, n_components=n_components, test_round=8)
			# Store in



# Hacker / Fair
hackerargs  = {'id':76561197979652439 ,'class':'hacker', 'start_tick':0 , 'end_tick':1000000, "enemy_name":"Eugene"}
dfhacker = data_munge("32t_de_dust2_hack_2.csv", "32t_de_dust2_hack_2_attackinfo.csv", dictargs=hackerargs)

fairargs = {'id':76561197979669175, 'class':'fair', 'start_tick':60000, 'end_tick':160000, "enemy_name": "ENVYUS HappyV"}
dffair = data_munge("128t_de_inferno_186_envyus-dignitas_de_inferno.csv", "128t_de_inferno_186_envyus-dignitas_de_inferno_attackinfo.csv", dictargs=fairargs)

# using flusha's id
suspectargs = {'id':76561197991348083 ,'class':'hacker', 'start_tick':310500 , 'end_tick':317000, "enemy_name":"Xizt Xtrfy"}
dfsuspect = data_munge("128t_de_inferno_2013_fnatic-ninjas-in-pyjamas_de_inferno.csv", "128t_de_inferno_2013_fnatic-ninjas-in-pyjamas_de_inferno_attackinfo.csv", dictargs=suspectargs)

classify_analysis(dfhacker, dffair, ["ViewDiffSpeed","AimbotDist"], n_components=4,window_size=128,custom_test_data=dfsuspect)
classify_analysis(dfhacker, dffair, ["ViewDiffSpeed"], n_components=4,window_size=256,custom_test_data=dfsuspect)
# classify_analysis(dfhacker, dffair, ["TrueViewRadDiffSpeed", "TrueViewDiffSpeed"], n_components=4,window_size=128*2)
# classify_analysis(dfhacker, dffair, ["ViewRad"], n_components=8,window_size=128)



## Ensemble
#dfhacker_new, dffair_new = classify_analysis(dfhacker, dffair, ["ViewRadDiffSpeed"], n_components=8,window_size=128, drawplot=False)
#dfhacker_new, dffair_new = classify_analysis(dfhacker_new, dffair_new, ["ViewDiffSpeed"], n_components=8 , window_size=128, drawplot=False)
# Cool so we have the data we need, but the issue is that we don't even care about the data...
# we care about having the ViewRadDiffSpeed Model.. and the ViewDiffSpeed... 
# We then predict the data
pdb.set_trace()
#dffhacker, dffair =classify_analysis(dfhacker_new, dffair_new, ["HMM_ViewDiffSpeed", "HMM_ViewRadDiffSpeed" ], n_components=8, window_size=128)

#classify_analysis(dfhacker, dffair, columns)

# Pass data in df, along with model:
# predict data with model, and return simplified states..
# Then given simplified states create a new column.. 
# then simply call the data with.. new columns.., and 
# classify_analysis(moddfhacker, moddffair, ["HMM_ViewSpeed", "HMM_RadSpeed"])







# Plot X / Y corelation of score...

pdb.set_trace()
print "stop"





