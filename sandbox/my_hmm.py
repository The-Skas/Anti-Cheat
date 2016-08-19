from data_helpers import *

from hmmlearn.hmm import GaussianHMM

import pandas as pd
import sys
import pdb
from sklearn.externals import joblib
#plot
import csgo_plot
import time
# Split rounds into multiple dataframes.

def split_rounds(df):
	rounds = df.Round.unique()
	dictrounds = {elem : pd.DataFrame for elem in rounds}
	for key in dictrounds.keys():
		# Drops columns if any missing value occurs:
		# POSSIBLE ISSUES!!! Breaks sequence of states... so that
		# The transition to a state could be false..
		_df = df[:][df.Round == key].dropna()
		_length = len(_df)
		dictrounds[key] = (_df , _length)
	return dictrounds

def predict_rounds_markov_model(model, columns, df):
	dfclean = df.dropna(subset=columns)
	dfclean = dfclean.dropna()
	dictrounds = split_rounds(dfclean)

	_df = None 
	for key in dictrounds.keys():
		# Get at index 0 since its tupled of df
		S = dictrounds[key][0][columns].as_matrix()
		X = np.vstack([S])
		dictrounds[key][0]["HMM_"+"".join(columns)] = model.predict(X)
		# Must set 'is' here as pandas uses boolean for masking.
		if _df is None: 
			_df = dictrounds[key][0]["HMM_"+"".join(columns)] 
		else:
			_df = pd.concat([_df, dictrounds[key][0]["HMM_"+"".join(columns)]]) 

	return df.join(_df)


def create_markov_model(df, columns, n_components, covariance_type="diag", n_iter=1000,test_round=None,dictrounds=None):
	# Given a data frame, list of columns for the markov model

	# If we didnt pass our own rounds.. 
	if not dictrounds:
		# drop all data which has any of the columns null.
		#remove null values
		dfclean = df.dropna(subset=columns)
		dfclean = dfclean.dropna()
		# Get rounds
		dictrounds = split_rounds(dfclean)

	import random
	_test_round = None
	if(test_round == "r"):
		rand_round = random.choice(dictrounds.keys())
		_test_round = dictrounds.pop(rand_round)
	elif(test_round):
		_test_round = dictrounds.pop(test_round)


	# Get Appropriate Training Dimensions
	X = np.reshape(np.array([]),(0,len(columns)))

	for key in dictrounds.keys():
		# Get at index 0 since its tupled of df
		S = dictrounds[key][0][columns].as_matrix()
		X = np.vstack([X,S])

	lengths = [ dictrounds[key][1] for key in dictrounds.keys()]
	model = GaussianHMM(n_components= n_components, covariance_type= covariance_type , n_iter= n_iter).fit(X, lengths)

	joblib.dump(model, "models/"+'_'.join(columns)+".pkl")

	if test_round:
		return model, X, _test_round
	else:
		return model, X




import time
def main():
	time_1 = time.time()

	#id class start_tick end_tick
	dictargs = dict([arg.split('=') for arg in sys.argv[3:]])
	pdb.set_trace()
	dfplayer = clean_data_to_numbers(sys.argv[1], sys.argv[2], dictargs=dictargs)

	# Get unique ID... Hmmmm 
	# Hm the real question is to create a classifier for now.
	print "Done Parsing: {:0.5f}".format(time.time() - time_1)
	plt.show(block=False)

	# Chose 16 components due to dividing around a circle.
	time_1 = time.time()
	model_1, X_1 = create_markov_model(dfplayer, ["TrueViewSin","TrueViewCos"], n_components = 8)
	print "Done markov model 1:{:0.5f}".format(time.time() - time_1)

	# Plot as an Axis Around the model.
	csgo_plot.plot_scatter_hmm(X_1[:,0] ,X_1[:,1] , model_1, X_1, title=dictargs["class"] + "-- Sin | Cos ")

	# plt.show()

	#Chose 4 components to give some variance.
	time_1 = time.time()
	model_2, X_2 = create_markov_model(dfplayer, ["TrueViewDiff"] , n_components=4)
	print "Done markov model 2: {:0.5f}".format(time.time() - time_1)

	# Plot as a linegraph over time.
	csgo_plot.plot_plane(np.arange(len(X_2)), X_2, model_2, X_2, title=dictargs["class"]+" -- TrueViewDiff")
	plt.show(block=False)


	model_d, X_d = create_markov_model(dfplayer, ["TrueViewDiffSpeed"], n_components=4)
	csgo_plot.plot_plane(np.arange(len(X_d)), X_d, model_d, X_d, title=dictargs["class"]+" -- TrueViewDiffSpeed")

	plt.show(block=False)

	model_r, X_r = create_markov_model(dfplayer, ["TrueViewRadDiff"], n_components=8)
	csgo_plot.plot_plane(np.arange(len(X_r)), X_r, model_r, X_r, title=dictargs["class"]+" -- TrueViewRadDiff")

	plt.show(block=False)
	# time_1 = time.time()

	# model_3, X_3 = create_markov_model(dfplayer, ["TrueViewDiff","TrueViewSin","TrueViewCos"] , n_components=8)
	# print "Done markov model 2: {:0.5f}".format(time.time() - time_1)

	# csgo_plot.plot_scatter_hmm(X_3[:,1] ,X_3[:,2] , model_3, X_3, title=dictargs["class"]+" -- TrueViewDiff | Sin | Cos ")
	# plt.show(block=False)

	# model_4, X_4 = create_markov_model(dfplayer, ["TrueViewRadDiff","TrueViewDiff"], n_components = 8)
	# csgo_plot.plot_scatter_hmm(X_3[:,1] ,X_3[:,2] , model_3, X_3, title=dictargs["class"]+" -- TrueViewRad | TrueViewDiff ")

	pdb.set_trace()
	## Get all Non-Null ViewDiffs and Aimbotdist
	## Issue with this is the fucking up the time series...

	write_model("data/RandomForestModel.csv", result, test_passenger_id)


	# Do score here.
	print "Done!"

if __name__ == '__main__':
	main()

